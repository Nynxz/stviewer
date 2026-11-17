//! Colours, spacing and small reusable widgets (cards, chips, badges).

use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Margin, Response, RichText, Shadow, Stroke,
    TextStyle, Theme, ThemePreference, Ui, Visuals,
};
use stv_core::Dtype;

pub struct Palette {
    pub bg: Color32,
    pub surface: Color32,
    pub surface_alt: Color32,
    pub input: Color32,
    pub border: Color32,
    pub border_strong: Color32,
    pub hover: Color32,
    pub text: Color32,
    pub text_weak: Color32,
    pub accent: Color32,
    pub accent_soft: Color32,
    pub on_accent: Color32,
    pub warn: Color32,
    pub error: Color32,
    pub success: Color32,
}

pub const DARK: Palette = Palette {
    bg: Color32::from_rgb(0x12, 0x14, 0x18),
    surface: Color32::from_rgb(0x1a, 0x1d, 0x23),
    surface_alt: Color32::from_rgb(0x21, 0x25, 0x2c),
    input: Color32::from_rgb(0x0e, 0x10, 0x13),
    border: Color32::from_rgb(0x2a, 0x2f, 0x38),
    border_strong: Color32::from_rgb(0x3a, 0x41, 0x4d),
    hover: Color32::from_rgb(0x28, 0x2d, 0x35),
    text: Color32::from_rgb(0xe3, 0xe6, 0xeb),
    text_weak: Color32::from_rgb(0x8b, 0x93, 0xa1),
    accent: Color32::from_rgb(0xf5, 0xa5, 0x24),
    accent_soft: Color32::from_rgb(0x3a, 0x2e, 0x17),
    on_accent: Color32::from_rgb(0x1a, 0x12, 0x02),
    warn: Color32::from_rgb(0xf0, 0xb4, 0x4c),
    error: Color32::from_rgb(0xf0, 0x6a, 0x6a),
    success: Color32::from_rgb(0x4c, 0xc3, 0x8a),
};

pub const LIGHT: Palette = Palette {
    bg: Color32::from_rgb(0xf4, 0xf5, 0xf7),
    surface: Color32::from_rgb(0xff, 0xff, 0xff),
    surface_alt: Color32::from_rgb(0xee, 0xf0, 0xf3),
    input: Color32::from_rgb(0xfb, 0xfb, 0xfc),
    border: Color32::from_rgb(0xe0, 0xe3, 0xe8),
    border_strong: Color32::from_rgb(0xc9, 0xce, 0xd6),
    hover: Color32::from_rgb(0xe8, 0xeb, 0xef),
    text: Color32::from_rgb(0x1d, 0x21, 0x27),
    text_weak: Color32::from_rgb(0x6a, 0x72, 0x80),
    accent: Color32::from_rgb(0xd0, 0x80, 0x0c),
    accent_soft: Color32::from_rgb(0xfd, 0xf0, 0xd9),
    on_accent: Color32::from_rgb(0xff, 0xff, 0xff),
    warn: Color32::from_rgb(0xb4, 0x6a, 0x00),
    error: Color32::from_rgb(0xc8, 0x32, 0x32),
    success: Color32::from_rgb(0x1f, 0x8a, 0x55),
};

pub fn palette(ui: &Ui) -> &'static Palette {
    if ui.visuals().dark_mode {
        &DARK
    } else {
        &LIGHT
    }
}

pub fn install(ctx: &egui::Context) {
    ctx.set_theme(ThemePreference::System);
    ctx.set_visuals_of(Theme::Dark, visuals(&DARK, true));
    ctx.set_visuals_of(Theme::Light, visuals(&LIGHT, false));
    ctx.all_styles_mut(|style| {
        use FontFamily::{Monospace, Proportional};
        style.text_styles = [
            (TextStyle::Small, FontId::new(11.5, Proportional)),
            (TextStyle::Body, FontId::new(14.0, Proportional)),
            (TextStyle::Button, FontId::new(14.0, Proportional)),
            (TextStyle::Heading, FontId::new(22.0, Proportional)),
            (TextStyle::Monospace, FontId::new(13.0, Monospace)),
        ]
        .into();
        let s = &mut style.spacing;
        s.item_spacing = egui::vec2(8.0, 6.0);
        s.button_padding = egui::vec2(10.0, 5.0);
        s.interact_size.y = 26.0;
        s.menu_margin = Margin::same(6);
        s.window_margin = Margin::same(16);
        s.scroll.floating = false;
        // Subtle handle: a muted fill rather than the full text colour, brighter on hover.
        s.scroll.foreground_color = false;
        s.scroll.bar_width = 6.0;
        s.scroll.bar_inner_margin = 4.0;
        s.scroll.handle_min_length = 24.0;
        s.scroll.dormant_background_opacity = 0.0;
        s.scroll.active_background_opacity = 0.0;
        s.scroll.interact_background_opacity = 0.0;
        s.scroll.dormant_handle_opacity = 1.0;
        s.scroll.active_handle_opacity = 1.0;
        s.scroll.interact_handle_opacity = 1.0;
    });
}

fn visuals(p: &Palette, dark: bool) -> Visuals {
    let mut v = if dark {
        Visuals::dark()
    } else {
        Visuals::light()
    };
    let r = CornerRadius::same(6);

    v.panel_fill = p.bg;
    v.window_fill = p.surface;
    v.extreme_bg_color = p.input;
    v.text_edit_bg_color = Some(p.input);
    v.faint_bg_color = p.surface_alt;
    v.code_bg_color = p.surface_alt;
    v.window_stroke = Stroke::new(1.0, p.border);
    v.window_corner_radius = CornerRadius::same(12);
    v.menu_corner_radius = CornerRadius::same(8);
    v.window_shadow = Shadow {
        offset: [0, 8],
        blur: 28,
        spread: 0,
        color: Color32::from_black_alpha(90),
    };
    v.popup_shadow = Shadow {
        offset: [0, 4],
        blur: 14,
        spread: 0,
        color: Color32::from_black_alpha(60),
    };
    v.hyperlink_color = p.accent;
    v.warn_fg_color = p.warn;
    v.error_fg_color = p.error;
    v.selection.bg_fill = p.accent.gamma_multiply(if dark { 0.35 } else { 0.3 });
    v.selection.stroke = Stroke::new(1.0, p.accent);
    v.text_cursor.stroke = Stroke::new(2.0, p.accent);
    v.weak_text_color = Some(p.text_weak);
    v.interact_cursor = Some(egui::CursorIcon::PointingHand);
    v.striped = false;

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = p.surface;
    w.noninteractive.weak_bg_fill = p.surface;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.text);
    w.noninteractive.corner_radius = r;

    // `bg_fill` (not `weak_bg_fill`) is what scrollbar handles and checkboxes use.
    w.inactive.bg_fill = p.border_strong;
    w.inactive.weak_bg_fill = p.surface_alt;
    w.inactive.bg_stroke = Stroke::new(1.0, p.border);
    w.inactive.fg_stroke = Stroke::new(1.0, p.text);
    w.inactive.corner_radius = r;

    w.hovered.bg_fill = p.text_weak;
    w.hovered.weak_bg_fill = p.hover;
    w.hovered.bg_stroke = Stroke::new(1.0, p.border_strong);
    w.hovered.fg_stroke = Stroke::new(1.5, p.text);
    w.hovered.corner_radius = r;
    w.hovered.expansion = 0.0;

    w.active.bg_fill = p.text_weak;
    w.active.weak_bg_fill = p.border;
    w.active.bg_stroke = Stroke::new(1.0, p.accent);
    w.active.fg_stroke = Stroke::new(1.5, p.text);
    w.active.corner_radius = r;
    w.active.expansion = 0.0;

    w.open = w.hovered;
    v
}

/// Rounded, bordered container used for every content block.
pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let p = palette(ui);
    egui::Frame::new()
        .fill(p.surface)
        .stroke(Stroke::new(1.0, p.border))
        .corner_radius(10)
        .inner_margin(Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// Small uppercase caption above a card's content.
pub fn caption(ui: &mut Ui, text: &str) {
    ui.label(
        RichText::new(text.to_uppercase())
            .size(11.0)
            .color(palette(ui).text_weak)
            .strong(),
    );
    ui.add_space(2.0);
}

/// Pill with tinted background, e.g. a dtype or a status badge.
pub fn chip(ui: &mut Ui, text: impl Into<String>, color: Color32, mono: bool) -> Response {
    let tint = if ui.visuals().dark_mode { 0.16 } else { 0.12 };
    let mut rt = RichText::new(text.into()).size(11.5).color(color);
    if mono {
        rt = rt.monospace();
    }
    egui::Frame::new()
        .fill(color.gamma_multiply(tint))
        .corner_radius(4)
        .inner_margin(Margin::symmetric(6, 1))
        .show(ui, |ui| ui.label(rt))
        .response
}

pub fn dtype_chip(ui: &mut Ui, dtype: &Dtype) -> Response {
    let c = dtype_color(dtype, ui.visuals().dark_mode);
    chip(ui, dtype.as_str(), c, true)
}

pub fn dtype_color(dtype: &Dtype, dark: bool) -> Color32 {
    let (d, l) = match dtype {
        Dtype::F64 | Dtype::F32 => ((0x5b, 0x9b, 0xff), (0x1f, 0x63, 0xd6)),
        Dtype::F16 => ((0x4c, 0xc3, 0x8a), (0x1a, 0x8a, 0x55)),
        Dtype::BF16 => ((0xb0, 0x8a, 0xf7), (0x7a, 0x4c, 0xd9)),
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::F8E8M0
        | Dtype::F6E2M3
        | Dtype::F6E3M2
        | Dtype::F4 => ((0xf5, 0x8f, 0x52), (0xc9, 0x5a, 0x16)),
        Dtype::C64 => ((0x4f, 0xc8, 0xd0), (0x13, 0x8a, 0x93)),
        Dtype::Other(_) => ((0xe0, 0x6c, 0x9f), (0xb8, 0x33, 0x6d)),
        _ => ((0x9a, 0xa4, 0xb5), (0x5c, 0x66, 0x77)),
    };
    let (r, g, b) = if dark { d } else { l };
    Color32::from_rgb(r, g, b)
}

/// Accent-filled button for the primary action.
pub fn primary_button(ui: &mut Ui, enabled: bool, text: &str) -> Response {
    let p = palette(ui);
    let btn = if enabled {
        egui::Button::new(RichText::new(text).color(p.on_accent).strong())
            .fill(p.accent)
            .stroke(Stroke::NONE)
    } else {
        egui::Button::new(text)
    };
    ui.add_enabled(enabled, btn)
}

/// Borderless button that only shows a background on hover.
pub fn ghost_button(ui: &mut Ui, text: impl Into<RichText>) -> Response {
    ui.add(
        egui::Button::new(text.into())
            .fill(Color32::TRANSPARENT)
            .stroke(Stroke::NONE),
    )
}

/// Segmented control: a row of mutually exclusive options.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, String)]) {
    let p = palette(ui);
    egui::Frame::new()
        .fill(p.surface_alt)
        .corner_radius(8)
        .inner_margin(Margin::same(3))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.horizontal(|ui| {
                for (opt, label) in options {
                    let sel = *value == *opt;
                    let text = RichText::new(label).color(if sel { p.text } else { p.text_weak });
                    let btn = egui::Button::new(if sel { text.strong() } else { text })
                        .fill(if sel { p.surface } else { Color32::TRANSPARENT })
                        .stroke(if sel {
                            Stroke::new(1.0, p.border)
                        } else {
                            Stroke::NONE
                        })
                        .corner_radius(6);
                    if ui.add(btn).clicked() {
                        *value = *opt;
                    }
                }
            });
        });
}

/// The app mark: an accent diamond with a dark core, painted (no font glyph needed).
pub fn brand_mark(ui: &mut Ui, size: f32) {
    let p = palette(ui);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let c = rect.center();
    let r = size * 0.48;
    let diamond = |r: f32| {
        vec![
            c + egui::vec2(0.0, -r),
            c + egui::vec2(r, 0.0),
            c + egui::vec2(0.0, r),
            c + egui::vec2(-r, 0.0),
        ]
    };
    let painter = ui.painter();
    painter.add(egui::Shape::convex_polygon(
        diamond(r),
        p.accent,
        Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        diamond(r * 0.42),
        p.surface,
        Stroke::NONE,
    ));
}

/// Window icon pixels (same mark as [`brand_mark`]), generated so no image asset is needed.
pub fn window_icon() -> egui::IconData {
    const N: u32 = 64;
    let mut rgba = Vec::with_capacity((N * N * 4) as usize);
    let half = N as f32 / 2.0;
    for y in 0..N {
        for x in 0..N {
            // Manhattan distance gives a diamond; 1px soft edge for anti-aliasing.
            let d = ((x as f32 + 0.5 - half).abs() + (y as f32 + 0.5 - half).abs()) / half;
            let outer = (0.96 - d).clamp(0.0, 0.03) / 0.03;
            let inner = (0.40 - d).clamp(0.0, 0.03) / 0.03;
            let (r, g, b) = if inner > 0.5 {
                (0x1a, 0x1d, 0x23)
            } else {
                (0xf5, 0xa5, 0x24)
            };
            rgba.extend_from_slice(&[r, g, b, (outer * 255.0) as u8]);
        }
    }
    egui::IconData {
        rgba,
        width: N,
        height: N,
    }
}
