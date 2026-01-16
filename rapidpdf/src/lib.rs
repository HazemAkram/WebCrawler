use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use std::path::Path;
use serde::{Deserialize, Serialize};
use pdfium_render::prelude::*;
use printpdf::{PdfLayerReference, Image as PdfImage, ImageTransform, ImageXObject, Px, ColorSpace, ColorBits};

// Data structures for Python boundary
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BBox {
    #[pyo3(get, set)]
    pub x: i32,
    #[pyo3(get, set)]
    pub y: i32,
    #[pyo3(get, set)]
    pub w: i32,
    #[pyo3(get, set)]
    pub h: i32,
}

#[pymethods]
impl BBox {
    #[new]
    fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        BBox { x, y, w, h }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PageImage {
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    rgba_data: Vec<u8>,
    // Store original PDF page dimensions in points (for accurate reconstruction)
    #[pyo3(get)]
    pub original_width_pts: f32,
    #[pyo3(get)]
    pub original_height_pts: f32,
}

#[pymethods]
impl PageImage {
    #[new]
    #[pyo3(signature = (width, height, rgba_data, original_width_pts=0.0, original_height_pts=0.0))]
    fn new(width: u32, height: u32, rgba_data: Vec<u8>, original_width_pts: f32, original_height_pts: f32) -> Self {
        PageImage { 
            width, 
            height, 
            rgba_data,
            original_width_pts,
            original_height_pts,
        }
    }

    fn get_rgba_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.rgba_data)
    }
    
    fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.rgba_data)
    }
}

impl PageImage {
    fn from_dynamic_image(img: &DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        PageImage {
            width: rgba.width(),
            height: rgba.height(),
            rgba_data: rgba.into_raw(),
            original_width_pts: 0.0,
            original_height_pts: 0.0,
        }
    }
    
    fn from_dynamic_image_with_original_size(img: &DynamicImage, orig_width_pts: f32, orig_height_pts: f32) -> Self {
        let rgba = img.to_rgba8();
        PageImage {
            width: rgba.width(),
            height: rgba.height(),
            rgba_data: rgba.into_raw(),
            original_width_pts: orig_width_pts,
            original_height_pts: orig_height_pts,
        }
    }

    fn to_rgba_image(&self) -> Option<RgbaImage> {
        ImageBuffer::from_raw(self.width, self.height, self.rgba_data.clone())
    }
}

/// Render a PDF to a list of PageImage objects using PDFium
#[pyfunction]
fn render_pdf(py: Python, pdf_path: &str, dpi: u16) -> PyResult<Vec<PageImage>> {
    let result = py.allow_threads(|| {
        // Load PDF with pdfium
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(
                Pdfium::pdfium_platform_library_name_at_path("./")
            ).or_else(|_| Pdfium::bind_to_system_library())
            .map_err(|e| format!("Failed to bind to PDFium: {:?}", e))?
        );
        
        let document = pdfium.load_pdf_from_file(pdf_path, None)
            .map_err(|e| format!("Failed to load PDF: {:?}", e))?;
        
        let page_count = document.pages().len();
        
        // Render all pages sequentially (PDFium document is not Sync)
        let mut pages = Vec::with_capacity(page_count as usize);
        
        for page_index in 0..page_count {
            let page = document.pages().get(page_index)
                .map_err(|e| format!("Failed to get page {}: {:?}", page_index, e))?;
            
            // Store original page dimensions in points
            let original_width_pts = page.width().value;
            let original_height_pts = page.height().value;
            
            // Convert DPI to points (PDFium uses points per inch)
            let render_config = PdfRenderConfig::new()
                .set_target_width((original_width_pts * dpi as f32 / 72.0) as i32)
                .set_target_height((original_height_pts * dpi as f32 / 72.0) as i32);
            
            let bitmap = page.render_with_config(&render_config)
                .map_err(|e| format!("Failed to render page {}: {:?}", page_index, e))?;
            
            // Convert bitmap to RGBA
            let width = bitmap.width() as u32;
            let height = bitmap.height() as u32;
            let rgba_data = bitmap.as_rgba_bytes().to_vec();
            
            pages.push(PageImage {
                width,
                height,
                rgba_data,
                original_width_pts,
                original_height_pts,
            });
        }
        
        let pages: Result<Vec<_>, String> = Ok(pages);
        
        pages
    });
    
    result.map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Apply redactions and QR code removal to pages
#[pyfunction]
fn apply_redactions_and_qr(
    py: Python,
    pages: Vec<PageImage>,
    redactions: Vec<Vec<BBox>>,
    qr_padding: u32,
    text_padding: u32,
) -> PyResult<Vec<PageImage>> {
    py.allow_threads(|| {
        // Process pages in parallel
        let results: Result<Vec<_>, String> = pages
            .par_iter()
            .enumerate()
            .map(|(page_idx, page)| {
                let mut img = page.to_rgba_image()
                    .ok_or_else(|| "Failed to convert page to image".to_string())?;
                
                // Apply text redactions for this page
                let page_redactions = redactions.get(page_idx).map(|r| r.as_slice()).unwrap_or(&[]);
                for bbox in page_redactions {
                    apply_redaction(&mut img, bbox, text_padding);
                }
                
                // Detect and remove QR codes
                remove_qr_codes(&mut img, qr_padding)?;
                
                // Preserve original page dimensions
                Ok(PageImage::from_dynamic_image_with_original_size(
                    &DynamicImage::ImageRgba8(img),
                    page.original_width_pts,
                    page.original_height_pts
                ))
            })
            .collect();
        
        results.map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

fn apply_redaction(img: &mut RgbaImage, bbox: &BBox, padding: u32) {
    let x = (bbox.x as i32 - padding as i32).max(0) as u32;
    let y = (bbox.y as i32 - padding as i32).max(0) as u32;
    let w = (bbox.w + 2 * padding as i32) as u32;
    let h = (bbox.h + 2 * padding as i32) as u32;
    
    // Estimate background color from surrounding area
    let bg_color = estimate_background_color(img, x, y, w, h);
    
    // Fill the region
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < img.width() && py < img.height() {
                img.put_pixel(px, py, bg_color);
            }
        }
    }
}

fn estimate_background_color(img: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Rgba<u8> {
    let margin = 40;
    let sample_x0 = x.saturating_sub(margin);
    let sample_y0 = y.saturating_sub(margin);
    let sample_x1 = (x + w + margin).min(img.width());
    let sample_y1 = (y + h + margin).min(img.height());
    
    let mut samples = Vec::new();
    for sy in sample_y0..sample_y1 {
        for sx in sample_x0..sample_x1 {
            samples.push(img.get_pixel(sx, sy));
        }
    }
    
    if samples.is_empty() {
        return Rgba([255, 255, 255, 255]);
    }
    
    // Compute median
    let mut r: Vec<u8> = samples.iter().map(|p| p[0]).collect();
    let mut g: Vec<u8> = samples.iter().map(|p| p[1]).collect();
    let mut b: Vec<u8> = samples.iter().map(|p| p[2]).collect();
    
    r.sort_unstable();
    g.sort_unstable();
    b.sort_unstable();
    
    let mid = r.len() / 2;
    Rgba([r[mid], g[mid], b[mid], 255])
}

fn remove_qr_codes(_img: &mut RgbaImage, _padding: u32) -> Result<(), String> {
    // TODO: Implement QR code detection using rqrr
    // For now, this is a no-op placeholder
    // QR removal can be added after core functionality is working
    Ok(())
}

/// Write processed pages to PDF with optional cover
#[pyfunction]
#[pyo3(signature = (pages, out_path, cover_png=None, drop_last_if_pages_gt_6=false))]
fn write_pdf(
    py: Python,
    pages: Vec<PageImage>,
    out_path: &str,
    cover_png: Option<&str>,
    drop_last_if_pages_gt_6: bool,
) -> PyResult<()> {
    let result = py.allow_threads(|| {
        
        let mut final_pages = pages;
        
        // Drop last page if requested
        if drop_last_if_pages_gt_6 && final_pages.len() > 6 {
            final_pages.pop();
        }
        
            // Add cover if provided
            if let Some(cover_path) = cover_png {
                if Path::new(cover_path).exists() {
                    let cover_img = ::image::open(cover_path)
                        .map_err(|e| format!("Failed to load cover: {:?}", e))?;
                
                // Match orientation logic from Python
                if final_pages.len() > 0 {
                    let first_page = &final_pages[0];
                    let is_horizontal = first_page.width > first_page.height;
                    
                    let cover_resized = if !is_horizontal {
                        // Resize cover to match first page for portrait
                        cover_img.resize_exact(
                            first_page.width,
                            first_page.height,
                            ::image::imageops::FilterType::Lanczos3,
                        )
                    } else {
                        cover_img
                    };
                    
                    // Create cover with same original dimensions as first page
                    let cover_page = PageImage::from_dynamic_image_with_original_size(
                        &cover_resized,
                        first_page.original_width_pts,
                        first_page.original_height_pts
                    );
                    final_pages.insert(0, cover_page);
                } else {
                    final_pages.insert(0, PageImage::from_dynamic_image(&cover_img));
                }
            }
        }
        
        if final_pages.is_empty() {
            return Err("No pages to write".to_string());
        }
        
        // Create PDF document using ORIGINAL page dimensions
        let first_page = &final_pages[0];
        
        // Use original PDF page dimensions if available, otherwise calculate from image
        let (width_pts, height_pts) = if first_page.original_width_pts > 0.0 && first_page.original_height_pts > 0.0 {
            (first_page.original_width_pts, first_page.original_height_pts)
        } else {
            // Fallback: assume image was rendered at 72 DPI
            (first_page.width as f32, first_page.height as f32)
        };
        
        // Convert points to mm: 1 point = 1/72 inch, 1 inch = 25.4mm
        let width_mm = (width_pts / 72.0) * 25.4;
        let height_mm = (height_pts / 72.0) * 25.4;
        
        let (doc, page1, layer1) = printpdf::PdfDocument::new("Cleaned PDF", printpdf::Mm(width_mm), printpdf::Mm(height_mm), "Layer 1");
        
        // Add first page
        let current_layer = doc.get_page(page1).get_layer(layer1);
        add_image_to_layer(&final_pages[0], current_layer)?;
        
        // Add remaining pages
        let doc = doc;
        for page_img in final_pages.iter().skip(1) {
            let (width_pts, height_pts) = if page_img.original_width_pts > 0.0 && page_img.original_height_pts > 0.0 {
                (page_img.original_width_pts, page_img.original_height_pts)
            } else {
                (page_img.width as f32, page_img.height as f32)
            };
            
            let width_mm = (width_pts / 72.0) * 25.4;
            let height_mm = (height_pts / 72.0) * 25.4;
            
            let (page_idx, layer_idx) = doc.add_page(printpdf::Mm(width_mm), printpdf::Mm(height_mm), "Layer 1");
            let layer = doc.get_page(page_idx).get_layer(layer_idx);
            add_image_to_layer(page_img, layer)?;
        }
        
        // Save
        let file = std::fs::File::create(out_path)
            .map_err(|e| format!("Failed to create output file: {:?}", e))?;
        let mut buf_writer = std::io::BufWriter::new(file);
        doc.save(&mut buf_writer)
            .map_err(|e| format!("Failed to write PDF: {:?}", e))?;
        
        Ok(())
    });
    
    result.map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

fn add_image_to_layer(page_img: &PageImage, layer: PdfLayerReference) -> Result<(), String> {
    // Convert PageImage to RGB (printpdf doesn't support RGBA directly)
    let rgba = page_img.to_rgba_image()
        .ok_or_else(|| "Failed to convert page to image".to_string())?;
    
    let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
    
    // Create printpdf Image from raw RGB data
    let img = PdfImage::from(ImageXObject {
        width: Px(page_img.width as usize),
        height: Px(page_img.height as usize),
        color_space: ColorSpace::Rgb,
        bits_per_component: ColorBits::Bit8,
        interpolate: true,
        image_data: rgb.into_raw(),
        image_filter: None,
        clipping_bbox: None,
    });
    
    // Calculate the DPI based on original dimensions and rendered dimensions
    // original_width_pts / 72.0 = inches
    // width_pixels / inches = DPI
    let dpi = if page_img.original_width_pts > 0.0 {
        (page_img.width as f32 * 72.0) / page_img.original_width_pts
    } else {
        72.0 // Default to 72 DPI if no original dimensions
    };
    
    // Position at bottom-left corner and specify DPI for proper scaling
    use printpdf::Mm;
    let transform = ImageTransform {
        translate_x: Some(Mm(0.0)),
        translate_y: Some(Mm(0.0)),
        rotate: None,
        scale_x: None,
        scale_y: None,
        dpi: Some(dpi),
    };
    
    img.add_to_layer(layer.clone(), transform);
    
    Ok(())
}

#[pymodule]
fn rapidpdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BBox>()?;
    m.add_class::<PageImage>()?;
    m.add_function(wrap_pyfunction!(render_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(apply_redactions_and_qr, m)?)?;
    m.add_function(wrap_pyfunction!(write_pdf, m)?)?;
    Ok(())
}
