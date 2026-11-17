//! The three main tabs plus the empty/drop state.

use crate::doc::{Doc, EntryState, Node, SortKey, TensorView};
use crate::theme::{self, caption, card, chip, dtype_chip, dtype_color, ghost_button, palette};
use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Layout, Margin, RichText, Sense, Stroke,
    Ui, vec2,
};
use egui_extras::{Column, TableBuilder};
use serde_json::Value;
use stv_core::format;

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

/// Returns true when the "Open file" button is clicked.
pub fn empty_state(ui: &mut Ui, hovering_file: bool) -> bool {
    let p = palette(ui);
    let mut open = false;
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() * 0.5 - 150.0).max(24.0));
        let border = if hovering_file {
            p.accent
        } else {
            p.border_strong
        };
        egui::Frame::new()
            .fill(if hovering_file {
                p.accent_soft
            } else {
                p.surface
            })
            .stroke(Stroke::new(if hovering_file { 2.0 } else { 1.0 }, border))
            .corner_radius(16)
            .inner_margin(Margin::symmetric(56, 40))
            .show(ui, |ui| {
                ui.set_width(380.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("⬇").size(40.0).color(if hovering_file {
                        p.accent
                    } else {
                        p.text_weak
                    }));
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("Drop a .safetensors file")
                            .size(18.0)
                            .strong(),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("to inspect its metadata, tensors and precision")
                            .color(p.text_weak),
                    );
                    ui.add_space(18.0);
                    if theme::primary_button(ui, true, "  Open file…  ").clicked() {
                        open = true;
                    }
                    ui.add_space(4.0);
                    ui.label(RichText::new("Ctrl+O").small().color(p.text_weak));
                });
            });
        ui.add_space(16.0);
        ui.label(
            RichText::new("Only the header is read, tensor data is never loaded or executed")
                .small()
                .color(p.text_weak),
        );
    });
    open
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

const KNOWN_KEYS: &[(&str, &str)] = &[
    ("modelspec.title", "Title"),
    ("modelspec.architecture", "Architecture"),
    ("modelspec.author", "Author"),
    ("modelspec.description", "Description"),
    ("modelspec.license", "License"),
    ("modelspec.date", "Date"),
    ("ss_base_model_version", "Base model"),
    ("ss_sd_model_name", "Trained on"),
    ("ss_network_module", "Network"),
    ("ss_network_dim", "Rank"),
    ("ss_network_alpha", "Alpha"),
    ("ss_output_name", "Output name"),
    ("ss_num_epochs", "Epochs"),
    ("ss_training_comment", "Comment"),
    ("format", "Format"),
];

pub fn overview(ui: &mut Ui, doc: &mut Doc) {
    let p = palette(ui);
    let h = &doc.header;
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.add_space(4.0);
            // Title row
            ui.horizontal(|ui| {
                ui.label(RichText::new(doc.file_name()).heading().strong());
                for badge in badges(doc) {
                    chip(ui, badge, p.accent, false);
                }
            });
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(doc.path.display().to_string())
                        .small()
                        .color(p.text_weak),
                );
                if ghost_button(ui, RichText::new("📋 copy").small().color(p.text_weak)).clicked()
                {
                    ui.ctx().copy_text(doc.path.display().to_string());
                }
            });
            ui.add_space(10.0);

            // Stat cards
            let total = h.total_params();
            let stats = [
                ("Parameters", format::count(total), format!("{total}")),
                (
                    "File size",
                    format::bytes(h.file_len),
                    format!("header {}", format::bytes(h.header_len)),
                ),
                (
                    "Tensors",
                    h.tensors.len().to_string(),
                    format!("{} dtypes", h.dtype_breakdown().len()),
                ),
                ("Metadata", doc.meta.len().to_string(), "keys".into()),
            ];
            ui.columns(stats.len(), |cols| {
                for (ui, (label, value, sub)) in cols.iter_mut().zip(stats) {
                    card(ui, |ui| {
                        caption(ui, label);
                        ui.label(RichText::new(value).size(24.0).strong());
                        ui.label(RichText::new(sub).small().color(p.text_weak));
                    });
                }
            });
            ui.add_space(10.0);

            // Precision
            card(ui, |ui| {
                caption(ui, "Precision");
                precision_bar(ui, doc);
            });
            ui.add_space(10.0);

            // Model info + structure, side by side when there's room
            let info: Vec<(&str, &str)> = KNOWN_KEYS
                .iter()
                .filter_map(|(k, label)| {
                    h.metadata
                        .get(*k)
                        .and_then(Value::as_str)
                        .map(|v| (*label, v))
                })
                .filter(|(_, v)| !v.is_empty())
                .collect();
            let wide = ui.available_width() > 820.0;
            if wide && !info.is_empty() {
                ui.columns(2, |cols| {
                    card(&mut cols[0], |ui| model_info(ui, &info));
                    card(&mut cols[1], |ui| structure(ui, doc));
                });
            } else {
                if !info.is_empty() {
                    card(ui, |ui| model_info(ui, &info));
                    ui.add_space(10.0);
                }
                card(ui, |ui| structure(ui, doc));
            }

            if !doc.header.warnings.is_empty() {
                ui.add_space(10.0);
                card(ui, |ui| {
                    caption(ui, "Warnings");
                    for w in &doc.header.warnings {
                        ui.label(RichText::new(format!("⚠  {w}")).color(p.warn));
                    }
                });
            }
            ui.add_space(8.0);
        });
}

fn badges(doc: &Doc) -> Vec<&'static str> {
    let m = &doc.header.metadata;
    let mut out = Vec::new();
    if doc
        .header
        .tensors
        .iter()
        .any(|t| t.name.contains("lora_") || t.name.contains(".lora"))
    {
        out.push("LoRA");
    }
    if m.keys().any(|k| k.starts_with("ss_")) {
        out.push("kohya-ss");
    }
    if m.contains_key("workflow") || m.contains_key("prompt") {
        out.push("ComfyUI workflow");
    }
    if m.keys().any(|k| k.starts_with("modelspec.")) {
        out.push("modelspec");
    }
    out
}

fn precision_bar(ui: &mut Ui, doc: &Doc) {
    let p = palette(ui);
    let dark = ui.visuals().dark_mode;
    let h = &doc.header;
    let mut parts: Vec<_> = h
        .dtype_breakdown()
        .into_iter()
        .map(|(d, (n, params))| {
            let bytes: u64 = h
                .tensors
                .iter()
                .filter(|t| t.dtype == d)
                .map(|t| t.byte_len())
                .sum();
            (d, n, params, bytes)
        })
        .collect();
    parts.sort_by_key(|a| std::cmp::Reverse(a.3));
    let total = h.data_len().max(1) as f32;

    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 10.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 5.0, p.surface_alt);
    // Position each segment by cumulative share so the last one ends exactly at
    // the bar's right edge. Sub-pixel slivers (e.g. a few F32 norm weights) are
    // skipped here (they still appear in the legend) so they can't steal the
    // rounded end cap from the last visible segment.
    let mut segs = Vec::new();
    let mut cum = 0u64;
    for (d, _, _, bytes) in &parts {
        let x0 = rect.left() + rect.width() * cum as f32 / total;
        cum += bytes;
        let x1 = rect.left() + rect.width() * cum as f32 / total;
        if x1 - x0 >= 1.0 {
            segs.push((d, x0, x1));
        }
    }
    if let Some(last) = segs.last_mut() {
        last.2 = rect.right();
    }
    let n = segs.len();
    for (i, (d, x0, x1)) in segs.into_iter().enumerate() {
        let seg = egui::Rect::from_x_y_ranges(x0..=x1, rect.y_range());
        let (left, right) = (if i == 0 { 5 } else { 0 }, if i + 1 == n { 5 } else { 0 });
        let r = CornerRadius {
            nw: left,
            sw: left,
            ne: right,
            se: right,
        };
        painter.rect_filled(seg, r, dtype_color(d, dark));
    }
    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 18.0;
        for (d, n, params, bytes) in &parts {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (dot, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
                ui.painter()
                    .circle_filled(dot.center(), 4.5, dtype_color(d, dark));
                ui.label(RichText::new(d.as_str()).monospace().strong());
                ui.label(
                    RichText::new(format!(
                        "{:.1}%  ·  {} tensors  ·  {} params",
                        *bytes as f32 * 100.0 / total,
                        n,
                        format::count(*params)
                    ))
                    .color(p.text_weak),
                );
            });
        }
    });
}

fn model_info(ui: &mut Ui, info: &[(&str, &str)]) {
    let p = palette(ui);
    caption(ui, "Model info");
    egui::Grid::new("model_info")
        .num_columns(2)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            for (label, value) in info {
                ui.label(RichText::new(*label).color(p.text_weak));
                let flat: String = value
                    .chars()
                    .take(120)
                    .map(|c| if c.is_control() { ' ' } else { c })
                    .collect();
                ui.add(egui::Label::new(flat).truncate())
                    .on_hover_text(short(value, 2000));
                ui.end_row();
            }
        });
}

fn structure(ui: &mut Ui, doc: &Doc) {
    let p = palette(ui);
    caption(ui, "Structure");
    let total = doc.header.total_params().max(1) as f32;
    let shown = 8;
    for (name, count, params) in doc.structure.iter().take(shown) {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(RichText::new(name).monospace()).truncate());
        });
        ui.horizontal(|ui| {
            let frac = *params as f32 / total;
            let (rect, _) = ui.allocate_exact_size(
                vec2((ui.available_width() - 150.0).max(40.0), 6.0),
                Sense::hover(),
            );
            ui.painter().rect_filled(rect, 3.0, p.surface_alt);
            let mut fill = rect;
            fill.set_width((rect.width() * frac).max(2.0));
            ui.painter().rect_filled(fill, 3.0, p.accent);
            ui.label(
                RichText::new(format!("{} · {count} t", format::count(*params)))
                    .small()
                    .color(p.text_weak),
            );
        });
        ui.add_space(2.0);
    }
    if doc.structure.len() > shown {
        ui.label(
            RichText::new(format!("+ {} more groups", doc.structure.len() - shown))
                .small()
                .color(p.text_weak),
        );
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

enum MetaAction {
    Add,
    Delete(usize),
}

pub fn metadata(ui: &mut Ui, doc: &mut Doc) {
    let p = palette(ui);
    let mut action = None;

    egui::Panel::left("meta_keys")
        .resizable(true)
        .default_size(300.0)
        .size_range(200.0..=560.0)
        .frame(egui::Frame::new().inner_margin(Margin {
            right: 12,
            ..Default::default()
        }))
        .show(ui, |ui| {
            // Right-to-left: place the button first, then let the filter take exactly
            // the remaining width. Sizing it by hand (available - N) overflowed by a
            // few pixels, and a resizable panel grows to fit its content, so the
            // splitter crept outward on every repaint.
            let row_h = ui.spacing().interact_size.y;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), row_h),
                Layout::right_to_left(Align::Center),
                |ui| {
                    if ui
                        .button("+ Add")
                        .on_hover_text("Add a metadata key")
                        .clicked()
                    {
                        action = Some(MetaAction::Add);
                    }
                    ui.add_sized(
                        [ui.available_width(), row_h],
                        egui::TextEdit::singleline(&mut doc.meta_filter)
                            .hint_text("🔍 Filter keys"),
                    );
                },
            );
            ui.add_space(6.0);
            if doc.meta.is_empty() {
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("No metadata in this file").color(p.text_weak))
                });
            }
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    let filter = doc.meta_filter.trim().to_owned();
                    for i in 0..doc.meta.len() {
                        let (k, v) = &doc.meta[i];
                        if !filter.is_empty()
                            && !format::name_matches(k, &filter)
                            && !format::name_matches(prefix(v, SEARCH_LEN), &filter)
                        {
                            continue;
                        }
                        if key_row(ui, k, v, doc.selected == Some(i), doc.entry_state(i)).clicked()
                        {
                            doc.selected = Some(i);
                        }
                    }
                });
        });

    egui::CentralPanel::default()
        .frame(egui::Frame::new().inner_margin(Margin {
            left: 14,
            ..Default::default()
        }))
        .show(ui, |ui| {
            let Some(i) = doc.selected.filter(|&i| i < doc.meta.len()) else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new("Select a key to view or edit its value").color(p.text_weak),
                    )
                });
                return;
            };
            let state = doc.entry_state(i);
            let dup = doc
                .meta
                .iter()
                .enumerate()
                .any(|(j, (k, _))| j != i && *k == doc.meta[i].0);
            // Huge values (a hostile file can carry up to ~100 MiB of metadata)
            // would be re-laid-out and re-parsed every frame and freeze the UI,
            // so they open as a read-only preview unless the user opts in.
            let large = doc.meta[i].0.len() + doc.meta[i].1.len() > LARGE_VALUE
                && doc.edit_large != Some(i);
            let mut unlock = false;
            let (key, value) = &mut doc.meta[i];

            egui::Frame::new()
                .fill(p.surface)
                .stroke(Stroke::new(1.0, p.border))
                .corner_radius(10)
                .inner_margin(Margin::same(14))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    caption(ui, "Key");
                    if large {
                        ui.add(egui::Label::new(RichText::new(short(key, 300)).monospace().size(15.0)).truncate());
                    } else {
                        ui.add(
                            egui::TextEdit::singleline(key)
                                .font(FontId::monospace(15.0))
                                .desired_width(f32::INFINITY)
                                .margin(Margin::symmetric(8, 6)),
                        );
                    }
                    if key.is_empty() {
                        ui.label(
                            RichText::new("Key must not be empty")
                                .small()
                                .color(p.error),
                        );
                    } else if dup {
                        ui.label(
                            RichText::new("Another key has the same name")
                                .small()
                                .color(p.error),
                        );
                    }
                    ui.add_space(6.0);

                    let parsed = if large {
                        None
                    } else {
                        serde_json::from_str::<Value>(value)
                            .ok()
                            .filter(|v| v.is_object() || v.is_array())
                    };
                    ui.horizontal(|ui| {
                        let kind = match &parsed {
                            Some(Value::Object(o)) => format!("JSON object · {} fields", o.len()),
                            Some(Value::Array(a)) => format!("JSON array · {} items", a.len()),
                            _ => "text".into(),
                        };
                        chip(
                            ui,
                            kind,
                            if parsed.is_some() {
                                theme::dtype_color(&stv_core::Dtype::F32, ui.visuals().dark_mode)
                            } else {
                                p.text_weak
                            },
                            false,
                        );
                        if large {
                            chip(ui, format::bytes(value.len() as u64), p.warn, false);
                        } else {
                            chip(
                                ui,
                                format!("{} chars", value.chars().count()),
                                p.text_weak,
                                false,
                            );
                            let lines = value.lines().count();
                            if lines > 1 {
                                chip(ui, format!("{lines} lines"), p.text_weak, false);
                            }
                        }
                        match state {
                            EntryState::New => {
                                chip(ui, "new", p.success, false);
                            }
                            EntryState::Modified => {
                                chip(ui, "edited", p.accent, false);
                            }
                            EntryState::Unchanged => {}
                        }

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ghost_button(ui, RichText::new("🗑 Delete").color(p.error)).clicked()
                            {
                                action = Some(MetaAction::Delete(i));
                            }
                            if ghost_button(ui, "📋 Copy").clicked() {
                                ui.ctx().copy_text(value.clone());
                            }
                            if let Some(v) = &parsed {
                                if ghost_button(ui, "Minify").clicked() {
                                    *value = v.to_string();
                                }
                                if ghost_button(ui, "{ } Pretty").clicked() {
                                    *value = serde_json::to_string_pretty(v).unwrap_or_default();
                                }
                            }
                        });
                    });
                });
            ui.add_space(8.0);

            if large {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "⚠  This value is {}. Showing the first {} read-only; editing very large text can make the app unresponsive.",
                            format::bytes(value.len() as u64),
                            format::bytes(PREVIEW_LEN as u64)
                        ))
                        .color(p.warn),
                    );
                    if ui.button("Edit anyway").clicked() {
                        unlock = true;
                    }
                });
                ui.add_space(6.0);
                egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new(RichText::new(short(value, PREVIEW_LEN)).monospace()).wrap());
                    });
            } else {
                egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(value)
                                .code_editor()
                                .desired_width(f32::INFINITY)
                                .min_size(ui.available_size())
                                .margin(Margin::same(10)),
                        );
                    });
            }
            if unlock {
                doc.edit_large = Some(i);
            }
        });

    match action {
        Some(MetaAction::Add) => {
            let mut n = 1;
            let key = loop {
                let k = if n == 1 {
                    "new_key".to_owned()
                } else {
                    format!("new_key_{n}")
                };
                if !doc.meta.iter().any(|(e, _)| *e == k) {
                    break k;
                }
                n += 1;
            };
            doc.meta.push((key, String::new()));
            doc.selected = Some(doc.meta.len() - 1);
            doc.meta_filter.clear();
        }
        Some(MetaAction::Delete(i)) => {
            doc.meta.remove(i);
            doc.selected = if doc.meta.is_empty() {
                None
            } else {
                Some(i.min(doc.meta.len() - 1))
            };
        }
        None => {}
    }
}

/// Two-line list row: key, then a one-line preview of the value.
fn key_row(
    ui: &mut Ui,
    key: &str,
    value: &str,
    selected: bool,
    state: EntryState,
) -> egui::Response {
    let p = palette(ui);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 44.0), Sense::click());
    let painter = ui.painter_at(rect);
    if selected {
        painter.rect_filled(rect, 8.0, p.accent_soft);
        painter.rect_filled(
            egui::Rect::from_min_size(
                rect.left_top() + vec2(0.0, 10.0),
                vec2(3.0, rect.height() - 20.0),
            ),
            2.0,
            p.accent,
        );
    } else if resp.hovered() {
        painter.rect_filled(rect, 8.0, p.hover);
    }
    let text_w = rect.width() - 34.0;
    let short_key = short(key, 300);
    let key_text: &str = if key.is_empty() {
        "(empty key)"
    } else {
        &short_key
    };
    let g = painter.layout(
        key_text.to_owned(),
        FontId::monospace(13.0),
        if key.is_empty() { p.text_weak } else { p.text },
        f32::INFINITY,
    );
    let g = truncate_galley(
        ui,
        g,
        key_text,
        FontId::monospace(13.0),
        if key.is_empty() { p.text_weak } else { p.text },
        text_w,
    );
    painter.galley(rect.left_top() + vec2(12.0, 6.0), g, p.text);

    let preview: String = value
        .chars()
        .take(160)
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let preview = if preview.is_empty() {
        "(empty)".to_owned()
    } else {
        preview
    };
    let g = truncate_galley(
        ui,
        painter.layout_no_wrap(preview.clone(), FontId::proportional(11.5), p.text_weak),
        &preview,
        FontId::proportional(11.5),
        p.text_weak,
        text_w,
    );
    painter.galley(rect.left_top() + vec2(12.0, 25.0), g, p.text_weak);

    let dot = match state {
        EntryState::Unchanged => None,
        EntryState::Modified => Some(p.accent),
        EntryState::New => Some(p.success),
    };
    if let Some(c) = dot {
        painter.circle_filled(egui::pos2(rect.right() - 14.0, rect.center().y), 4.0, c);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(short(key, 500))
}

/// Lay out text on one line, cut with an ellipsis to fit `width`.
fn truncate_galley(
    ui: &Ui,
    full: std::sync::Arc<egui::Galley>,
    text: &str,
    font: FontId,
    color: Color32,
    width: f32,
) -> std::sync::Arc<egui::Galley> {
    if full.size().x <= width {
        return full;
    }
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_owned(), font, color);
    job.wrap = egui::text::TextWrapping::truncate_at_width(width);
    ui.fonts_mut(|f| f.layout_job(job))
}

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

pub fn tensors(ui: &mut Ui, doc: &mut Doc) {
    let p = palette(ui);
    ui.horizontal(|ui| {
        theme::segmented(
            ui,
            &mut doc.tensor_view,
            &[
                (TensorView::Tree, "Tree".into()),
                (TensorView::List, "List".into()),
            ],
        );
        ui.add_space(6.0);
        ui.add(
            egui::TextEdit::singleline(&mut doc.tensor_filter)
                .hint_text("🔍 Filter tensors (substring or * ? wildcards)")
                .desired_width(380.0),
        );
        doc.refresh();
        ui.label(
            RichText::new(format!(
                "{} of {}",
                doc.rows.len(),
                doc.header.tensors.len()
            ))
            .color(p.text_weak),
        );
    });
    ui.add_space(8.0);

    egui::Frame::new()
        .fill(p.surface)
        .stroke(Stroke::new(1.0, p.border))
        .corner_radius(10)
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            match doc.tensor_view {
                TensorView::List => tensor_list(ui, doc),
                TensorView::Tree => tensor_tree(ui, doc),
            }
        });
}

fn tensor_list(ui: &mut Ui, doc: &mut Doc) {
    let p = palette(ui);
    let tensors = &doc.header.tensors;
    let rows = &doc.rows;
    let mut sort = doc.sort;
    let row_h = 26.0;

    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .auto_shrink(false)
        .cell_layout(Layout::left_to_right(Align::Center))
        .column(Column::remainder().at_least(240.0).clip(true))
        .column(Column::initial(90.0).at_least(70.0))
        .column(Column::initial(160.0).at_least(80.0).clip(true))
        .column(Column::initial(90.0).at_least(60.0))
        .column(Column::initial(90.0).at_least(60.0))
        .header(28.0, |mut h| {
            for (label, key) in [
                ("Name", SortKey::Name),
                ("Dtype", SortKey::Dtype),
                ("Shape", SortKey::Name),
                ("Params", SortKey::Params),
                ("Size", SortKey::Size),
            ] {
                h.col(|ui| {
                    let sortable = label != "Shape";
                    let active = sortable && sort.0 == key;
                    let arrow = if active {
                        if sort.1 { " ⏶" } else { " ⏷" }
                    } else {
                        ""
                    };
                    let text = RichText::new(format!("{label}{arrow}"))
                        .small()
                        .strong()
                        .color(if active { p.text } else { p.text_weak });
                    if sortable {
                        if ui
                            .add(egui::Label::new(text).sense(Sense::click()))
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .clicked()
                        {
                            sort = if sort.0 == key {
                                (key, !sort.1)
                            } else {
                                (key, key == SortKey::Name || key == SortKey::Dtype)
                            };
                        }
                    } else {
                        ui.label(text);
                    }
                });
            }
        })
        .body(|body| {
            body.rows(row_h, rows.len(), |mut row| {
                let t = &tensors[rows[row.index()]];
                row.col(|ui| {
                    ui.add(
                        egui::Label::new(RichText::new(short(&t.name, 1000)).monospace())
                            .truncate(),
                    )
                    .on_hover_text(short(&t.name, 1000));
                });
                row.col(|ui| {
                    dtype_chip(ui, &t.dtype);
                });
                row.col(|ui| {
                    ui.label(
                        RichText::new(format::shape(&t.shape))
                            .monospace()
                            .color(p.text_weak),
                    );
                });
                row.col(|ui| {
                    ui.label(format::count(t.numel()));
                });
                row.col(|ui| {
                    ui.label(RichText::new(format::bytes(t.byte_len())).color(p.text_weak));
                });
            });
        });
    doc.sort = sort;
}

fn tensor_tree(ui: &mut Ui, doc: &Doc) {
    let filtering = !doc.tensor_filter.trim().is_empty();
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            // Open the first level by default when it's small, everything when filtering.
            let open_top = doc.tree.len() <= 8;
            for n in &doc.tree {
                tree_node(
                    ui,
                    doc,
                    n,
                    if filtering { true } else { open_top },
                    filtering,
                );
            }
        });
}

fn tree_node(ui: &mut Ui, doc: &Doc, node: &Node, default_open: bool, filtering: bool) {
    let p = palette(ui);
    if let Some(i) = node.tensor {
        let t = &doc.header.tensors[i];
        ui.horizontal(|ui| {
            ui.add_space(18.0);
            ui.label(RichText::new(short(&node.segment, 300)).monospace());
            dtype_chip(ui, &t.dtype);
            ui.label(
                RichText::new(format::shape(&t.shape))
                    .monospace()
                    .color(p.text_weak),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(format::bytes(t.byte_len()))
                        .small()
                        .color(p.text_weak),
                );
            });
        })
        .response
        .on_hover_text(short(&t.name, 1000));
        return;
    }
    let title = RichText::new(short(&node.segment, 300))
        .monospace()
        .strong();
    let resp = egui::CollapsingHeader::new(title)
        .id_salt(&node.path)
        .default_open(default_open)
        .open(if filtering { Some(true) } else { None })
        .show(ui, |ui| {
            let small = node.children.len() <= 4;
            for c in &node.children {
                tree_node(ui, doc, c, small, filtering);
            }
        });
    let r = resp.header_response.rect;
    ui.painter().text(
        egui::pos2(ui.max_rect().right() - 4.0, r.center().y),
        Align2::RIGHT_CENTER,
        format!(
            "{} params · {} tensors",
            format::count(node.params),
            node.count
        ),
        FontId::proportional(11.5),
        p.text_weak,
    );
}

// ---------------------------------------------------------------------------
// Bounding untrusted text
// ---------------------------------------------------------------------------

/// Values above this size open as a read-only preview.
const LARGE_VALUE: usize = 256 * 1024;
/// How much of a large value the preview shows.
const PREVIEW_LEN: usize = 64 * 1024;
/// How much of each value the key filter searches.
const SEARCH_LEN: usize = 64 * 1024;

/// The first `max` bytes of `s`, cut on a char boundary.
fn prefix(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Like [`prefix`] but marks the cut with an ellipsis. Used anywhere a string
/// from the file is laid out every frame (labels, tooltips), so a multi-MB
/// key or tensor name can't stall rendering.
fn short(s: &str, max: usize) -> std::borrow::Cow<'_, str> {
    if s.len() <= max {
        std::borrow::Cow::Borrowed(s)
    } else {
        std::borrow::Cow::Owned(format!("{}…", prefix(s, max)))
    }
}

#[cfg(test)]
mod tests {
    use super::{prefix, short};

    #[test]
    fn prefix_respects_char_boundaries() {
        let s = "aé日🚀";
        for max in 0..=s.len() {
            let p = prefix(s, max);
            assert!(s.starts_with(p) && p.len() <= max);
        }
        assert_eq!(short("abcdef", 3), "abc…");
        assert_eq!(short("abc", 3), "abc");
    }
}
