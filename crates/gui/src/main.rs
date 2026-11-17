#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod doc;
mod theme;
mod views;

use doc::Doc;
use eframe::egui::{
    self, Align, Key, KeyboardShortcut, Layout, Margin, Modifiers, RichText, Stroke,
};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;
use stv_core::{SafeTensorsHeader, WriteOptions, format, write_with_metadata};
use theme::palette;

fn main() -> eframe::Result {
    let initial = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("stviewer")
            .with_app_id("stviewer")
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([720.0, 460.0])
            .with_drag_and_drop(true)
            .with_icon(std::sync::Arc::new(theme::window_icon())),
        ..Default::default()
    };
    eframe::run_native(
        "stviewer",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx);
            let mut app = App::default();
            if let Some(p) = initial {
                app.open(p);
            }
            app.apply_dev_env(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Default, PartialEq, Clone, Copy)]
enum Tab {
    #[default]
    Overview,
    Metadata,
    Tensors,
}

enum Pending {
    Open(PathBuf),
    Close,
}

struct Status {
    text: String,
    error: bool,
}

struct SaveJob {
    rx: mpsc::Receiver<Result<SafeTensorsHeader, String>>,
    dst: PathBuf,
}

#[derive(Default)]
struct App {
    doc: Option<Doc>,
    tab: Tab,
    status: Option<Status>,
    save_job: Option<SaveJob>,
    confirm: Option<Pending>,
    backup: bool,
    allow_close: bool,
    title: String,
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

impl App {
    /// Debug builds only, for screenshots and UI work:
    /// - `STVIEWER_TAB=metadata|tensors|list` picks the starting view
    /// - `STVIEWER_KEY=<name>` selects a metadata key
    /// - `STVIEWER_THEME=dark|light` overrides the system theme
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    fn apply_dev_env(&mut self, ctx: &egui::Context) {
        #[cfg(debug_assertions)]
        {
            let var = |k: &str| std::env::var(k).ok();
            match var("STVIEWER_TAB").as_deref() {
                Some("metadata") => self.tab = Tab::Metadata,
                Some("tensors") => self.tab = Tab::Tensors,
                Some("list") => {
                    self.tab = Tab::Tensors;
                    if let Some(d) = &mut self.doc {
                        d.tensor_view = doc::TensorView::List;
                    }
                }
                _ => {}
            }
            if let Some(d) = &mut self.doc {
                d.selected = match var("STVIEWER_KEY") {
                    Some(k) => d.meta.iter().position(|(name, _)| *name == k),
                    None => (!d.meta.is_empty()).then_some(0),
                };
            }
            match var("STVIEWER_THEME").as_deref() {
                Some("dark") => ctx.set_theme(egui::ThemePreference::Dark),
                Some("light") => ctx.set_theme(egui::ThemePreference::Light),
                _ => {}
            }
        }
    }

    fn dirty(&self) -> bool {
        self.doc.as_ref().is_some_and(Doc::is_dirty)
    }

    fn saving(&self) -> bool {
        self.save_job.is_some()
    }

    fn set_status(&mut self, text: impl Into<String>, error: bool) {
        self.status = Some(Status {
            text: text.into(),
            error,
        });
    }

    fn request_open(&mut self, path: PathBuf) {
        if self.saving() {
            return;
        }
        if self.dirty() {
            self.confirm = Some(Pending::Open(path));
        } else {
            self.open(path);
        }
    }

    fn open(&mut self, path: PathBuf) {
        match Doc::load(path.clone()) {
            Ok(doc) => {
                let msg = match doc.header.warnings.len() {
                    0 => format!("Opened {}", doc.file_name()),
                    n => format!(
                        "Opened {} with {n} warning(s), see Overview",
                        doc.file_name()
                    ),
                };
                self.doc = Some(doc);
                self.set_status(msg, false);
            }
            Err(e) => self.set_status(format!("Could not open {}: {e}", path.display()), true),
        }
    }

    fn pick_file(&mut self) {
        let mut dialog = rfd::FileDialog::new().add_filter("safetensors", &["safetensors"]);
        if let Some(dir) = self.doc.as_ref().and_then(|d| d.path.parent()) {
            dialog = dialog.set_directory(dir);
        }
        if let Some(p) = dialog.pick_file() {
            self.request_open(p);
        }
    }

    fn save_as_dialog(&mut self) {
        let Some(doc) = &self.doc else { return };
        let stem = doc
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut dialog = rfd::FileDialog::new()
            .add_filter("safetensors", &["safetensors"])
            .set_file_name(format!("{stem}.edited.safetensors"));
        if let Some(dir) = doc.path.parent() {
            dialog = dialog.set_directory(dir);
        }
        if let Some(dst) = dialog.save_file() {
            self.save(Some(dst));
        }
    }

    /// Write edits in a background thread (tensor data may be gigabytes).
    fn save(&mut self, dst: Option<PathBuf>) {
        if self.saving() {
            return;
        }
        let Some(doc) = &self.doc else { return };
        let meta = match doc.edited_metadata() {
            Ok(m) => m,
            Err(e) => return self.set_status(format!("Can't save: {e}"), true),
        };
        let src = doc.path.clone();
        let dst = dst.unwrap_or_else(|| src.clone());
        let opts = WriteOptions {
            backup: self.backup,
        };
        let (tx, rx) = mpsc::channel();
        let thread_dst = dst.clone();
        std::thread::spawn(move || {
            let res =
                write_with_metadata(&src, &thread_dst, &meta, &opts).map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
        self.set_status(format!("Saving {}…", dst.display()), false);
        self.save_job = Some(SaveJob { rx, dst });
    }

    fn poll_save(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.save_job else { return };
        match job.rx.try_recv() {
            Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.save_job = None;
                self.set_status("Save failed: writer thread crashed", true);
            }
            Ok(res) => {
                let job = self.save_job.take().expect("checked above");
                match res {
                    Ok(header) => {
                        if let Some(doc) = &mut self.doc {
                            doc.reload(job.dst.clone(), header);
                        }
                        self.set_status(format!("Saved {}", job.dst.display()), false);
                    }
                    Err(e) => self.set_status(format!("Save failed: {e}"), true),
                }
            }
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path())
                .find(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
        });
        if let Some(p) = dropped {
            self.request_open(p);
        }

        let open = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
        let save = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
        let save_as = KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::S);
        let (o, sa, s) = ctx.input_mut(|i| {
            (
                i.consume_shortcut(&open),
                i.consume_shortcut(&save_as),
                i.consume_shortcut(&save),
            )
        });
        if o {
            self.pick_file();
        }
        if sa {
            self.save_as_dialog();
        } else if s && self.dirty() {
            self.save(None);
        }

        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.saving() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.set_status("Please wait for the save to finish", true);
            } else if self.dirty() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.confirm = Some(Pending::Close);
            }
        }
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        let title = match &self.doc {
            Some(d) => format!(
                "{}{} — stviewer",
                if d.is_dirty() { "● " } else { "" },
                d.file_name()
            ),
            None => "stviewer".into(),
        };
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_save(&ctx);
        self.handle_input(&ctx);
        self.update_title(&ctx);
        let p = palette(ui);
        let hovering_file = ctx.input(|i| !i.raw.hovered_files.is_empty());

        egui::Panel::top("toolbar")
            .frame(
                egui::Frame::new()
                    .fill(p.surface)
                    .inner_margin(Margin::symmetric(14, 8))
                    .stroke(Stroke::new(1.0, p.border)),
            )
            .show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status")
            .frame(
                egui::Frame::new()
                    .fill(p.surface)
                    .inner_margin(Margin::symmetric(14, 5))
                    .stroke(Stroke::new(1.0, p.border)),
            )
            .show(ui, |ui| self.status_bar(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(p.bg).inner_margin(Margin::same(16)))
            .show(ui, |ui| {
                let saving = self.saving();
                let tab = self.tab;
                let mut open_clicked = false;
                match &mut self.doc {
                    None => open_clicked = views::empty_state(ui, hovering_file),
                    Some(doc) => match tab {
                        Tab::Overview => views::overview(ui, doc),
                        Tab::Metadata => {
                            ui.add_enabled_ui(!saving, |ui| views::metadata(ui, doc));
                        }
                        Tab::Tensors => views::tensors(ui, doc),
                    },
                }
                if open_clicked {
                    self.pick_file();
                }
            });

        if hovering_file && self.doc.is_some() {
            drop_overlay(&ctx);
        }
        self.confirm_dialog(&ctx);
    }
}

impl App {
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let p = palette(ui);
        let saving = self.saving();
        let dirty = self.dirty();
        ui.horizontal(|ui| {
            theme::brand_mark(ui, 18.0);
            ui.label(RichText::new("stviewer").strong().size(15.0));

            if let Some(doc) = &self.doc {
                ui.add_space(12.0);
                let tabs = [
                    (Tab::Overview, "Overview".to_owned()),
                    (Tab::Metadata, format!("Metadata  {}", doc.meta.len())),
                    (Tab::Tensors, format!("Tensors  {}", doc.header.tensors.len())),
                ];
                theme::segmented(ui, &mut self.tab, &tabs);
            }

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button(RichText::new("☰").size(15.0), |ui| {
                    ui.set_min_width(200.0);
                    if ui.add_enabled(dirty && !saving, egui::Button::new("↺  Revert changes")).clicked() {
                        if let Some(d) = &mut self.doc {
                            d.reset_meta();
                        }
                        ui.close();
                    }
                    ui.checkbox(&mut self.backup, "Keep .bak when saving")
                        .on_hover_text("When saving over the original, keep the old file as <name>.bak (hard link, instant)");
                    ui.separator();
                    ui.label(RichText::new("Theme").small().color(p.text_weak));
                    let mut pref = ui.ctx().options(|o| o.theme_preference);
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut pref, egui::ThemePreference::System, "System");
                        ui.selectable_value(&mut pref, egui::ThemePreference::Dark, "Dark");
                        ui.selectable_value(&mut pref, egui::ThemePreference::Light, "Light");
                    });
                    ui.ctx().set_theme(pref);
                });

                if saving {
                    ui.spinner();
                } else {
                    let has_doc = self.doc.is_some();
                    if ui
                        .add_enabled(has_doc, egui::Button::new("Save as…"))
                        .on_hover_text("Ctrl+Shift+S")
                        .clicked()
                    {
                        self.save_as_dialog();
                    }
                    if theme::primary_button(ui, dirty, "Save").on_hover_text("Ctrl+S").clicked() {
                        self.save(None);
                    }
                }
                if ui.button("Open…").on_hover_text("Ctrl+O").clicked() {
                    self.pick_file();
                }
                if dirty && !saving {
                    ui.label(RichText::new("• unsaved").small().color(p.accent));
                }
            });
        });
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        let p = palette(ui);
        ui.horizontal(|ui| {
            if let Some(s) = &self.status {
                let (icon, color) = if s.error {
                    ("⚠", p.error)
                } else {
                    ("✔", p.success)
                };
                ui.label(RichText::new(icon).small().color(color));
                ui.label(RichText::new(&s.text).small().color(if s.error {
                    p.error
                } else {
                    p.text_weak
                }));
            }
            if let Some(doc) = &self.doc {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let h = &doc.header;
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!(
                                "{}   ·   {} params   ·   {}",
                                format::bytes(h.file_len),
                                format::count(h.total_params()),
                                doc.path.display()
                            ))
                            .small()
                            .color(p.text_weak),
                        )
                        .truncate(),
                    );
                });
            }
        });
    }

    fn confirm_dialog(&mut self, ctx: &egui::Context) {
        let Some(pending) = &self.confirm else { return };
        let what = match pending {
            Pending::Open(_) => "Open another file and discard your metadata edits?",
            Pending::Close => "Quit and discard your metadata edits?",
        };
        let mut choice = None;
        let resp = egui::Modal::new(egui::Id::unique("confirm_discard")).show(ctx, |ui| {
            let p = palette(ui);
            ui.set_width(340.0);
            ui.label(RichText::new("Unsaved changes").size(17.0).strong());
            ui.add_space(4.0);
            ui.label(RichText::new(what).color(p.text_weak));
            ui.add_space(14.0);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(
                        egui::Button::new(RichText::new("Discard").color(p.on_accent).strong())
                            .fill(p.error),
                    )
                    .clicked()
                {
                    choice = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    choice = Some(false);
                }
            });
        });
        if resp.should_close() && choice.is_none() {
            choice = Some(false);
        }
        match (choice, self.confirm.take()) {
            (Some(true), Some(Pending::Open(p))) => self.open(p),
            (Some(true), Some(Pending::Close)) => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            (None, p) => self.confirm = p,
            _ => {}
        }
    }
}

/// Full-window highlight while a file is dragged over an open document.
fn drop_overlay(ctx: &egui::Context) {
    let p = if ctx.global_style().visuals.dark_mode {
        &theme::DARK
    } else {
        &theme::LIGHT
    };
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::unique("drop_overlay"),
    ));
    let rect = ctx.content_rect().shrink(10.0);
    painter.rect(
        rect,
        14.0,
        p.bg.gamma_multiply(0.85),
        Stroke::new(2.0, p.accent),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "Drop to open",
        egui::FontId::proportional(22.0),
        p.accent,
    );
}
