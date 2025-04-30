import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from scipy.stats import skew, kurtosis, ttest_ind
from datetime import date

# === Constants ===
GRID_GUARDIANS_BLUE = "#00B7C3"  # Updated to match the color in the image
GRID_GUARDIANS_DARK = "#1A2A44"  # Dark background color from the image
GRID_GUARDIANS_LIGHT = "#2C3E50"  # Slightly lighter shade for panels
LOGO_PATH = r"F:\\Python\\Research Day 2025\\Grid Guardians.jpeg"
DEFAULT_DATASET_FILES = {
    'Train1': r'F:\\Python\\Research Day 2025\\Datasets\\Training\\signals_labels_binary_Tr0.csv',
    'Train2': r'F:\\Python\\Research Day 2025\\Datasets\\Training\\signals_labels_binary_Tr1.csv',
    'Validation': r'F:\\Python\\Research Day 2025\\Datasets\\Validation\\signals_labels_binary_Va0.csv',
}

# === Optimized Feature Extraction ===
def extract_features_vectorized(signal_series):
    signals = np.array([np.fromstring(s, sep=',', dtype=np.float32) for s in signal_series if isinstance(s, str) and s])
    
    if signals.size == 0:
        return pd.DataFrame(np.full((len(signal_series), 6), np.nan), columns=['mean', 'std', 'max', 'min', 'skew', 'kurtosis'])
    
    means = np.mean(signals, axis=1)
    stds = np.std(signals, axis=1)
    maxs = np.max(signals, axis=1)
    mins = np.min(signals, axis=1)
    skews = skew(signals, axis=1, nan_policy='omit')
    kurtoses = kurtosis(signals, axis=1, nan_policy='omit')
    
    return pd.DataFrame({
        'mean': means,
        'std': stds,
        'max': maxs,
        'min': mins,
        'skew': skews,
        'kurtosis': kurtoses
    })

def process_dataset(df):
    if 'Signal Values' not in df.columns or 'Label' not in df.columns:
        return None
    df = df.dropna(subset=['Signal Values'])
    if df.empty:
        return None
    features_df = extract_features_vectorized(df['Signal Values'])
    features_df['Label'] = df['Label'].astype(int).values
    return features_df.dropna()

# === GUI Application ===
class SignalsAnalysisApp:
    def _init_(self, root):
        self.root = root
        self.root.title("PD Signal Analysis Dashboard")
        self.root.geometry("800x400")
        self.root.configure(bg=GRID_GUARDIANS_DARK)

        # Main container
        main_frame = tk.Frame(self.root, bg=GRID_GUARDIANS_DARK)
        main_frame.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(main_frame, bg=GRID_GUARDIANS_LIGHT, width=150)
        sidebar.pack(side="left", fill="y")

        # Sidebar logo/icon
        try:
            if os.path.exists(LOGO_PATH):
                img = Image.open(LOGO_PATH)
                img = img.resize((40, 40), Image.LANCZOS)
                logo = ImageTk.PhotoImage(img)
                tk.Label(sidebar, image=logo, bg=GRID_GUARDIANS_LIGHT).pack(pady=10)
                sidebar.image = logo
        except Exception:
            tk.Label(sidebar, text="Logo Error", font=("Helvetica", 10), bg=GRID_GUARDIANS_LIGHT, fg="white").pack(pady=10)

        # Sidebar buttons (only Home and Load Data)
        tk.Button(sidebar, text="Home", font=("Helvetica", 12), bg=GRID_GUARDIANS_LIGHT, fg="white", bd=0, anchor="w", padx=20, pady=5).pack(fill="x")
        tk.Button(sidebar, text="Load Data", font=("Helvetica", 12), bg=GRID_GUARDIANS_BLUE, fg="white", bd=0, anchor="w", padx=20, pady=5).pack(fill="x")

        # Right side (header + content)
        right_frame = tk.Frame(main_frame, bg=GRID_GUARDIANS_DARK)
        right_frame.pack(side="left", fill="both", expand=True)

        # Header
        header = tk.Frame(right_frame, bg=GRID_GUARDIANS_DARK)
        header.pack(fill="x")
        tk.Label(header, text="PD Signal Analysis Dashboard", font=("Helvetica", 24, "bold"), bg=GRID_GUARDIANS_DARK, fg=GRID_GUARDIANS_BLUE).pack(pady=10)

        # Content area
        content = tk.Frame(right_frame, bg=GRID_GUARDIANS_DARK, padx=20, pady=20)
        content.pack(fill="both", expand=True)

        # Test file input
        tk.Label(content, text="Test File:", bg=GRID_GUARDIANS_DARK, fg="white", font=("Helvetica", 12)).pack(anchor="w")
        entry_frame = tk.Frame(content, bg=GRID_GUARDIANS_DARK)
        entry_frame.pack(anchor="w", pady=(5, 0))

        self.test_file_entry = tk.Entry(entry_frame, width=40, font=("Helvetica", 10), bg=GRID_GUARDIANS_LIGHT, fg="white", insertbackground="white")
        self.test_file_entry.pack(side="left", padx=(0, 5))
        tk.Button(entry_frame, text="Browse", command=self.browse_file, font=("Helvetica", 10), bg=GRID_GUARDIANS_BLUE, fg="white").pack(side="left")

        # Status label
        self.status_label = tk.Label(content, text="Ready", font=("Helvetica", 10), bg=GRID_GUARDIANS_DARK, fg="white")
        self.status_label.pack(anchor="w", pady=(20, 0))

        # Success message (mimicking the "Model trained successfully" in the image)
        success_frame = tk.Frame(content, bg=GRID_GUARDIANS_DARK)
        success_frame.pack(anchor="w", pady=(10, 0))
        tk.Label(success_frame, text="✔", font=("Helvetica", 12), bg=GRID_GUARDIANS_DARK, fg=GRID_GUARDIANS_BLUE).pack(side="left")
        tk.Label(success_frame, text="Model trained successfully", font=("Helvetica", 10), bg=GRID_GUARDIANS_DARK, fg="white").pack(side="left", padx=5)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.test_file_entry.delete(0, tk.END)
            self.test_file_entry.insert(0, path)

    def run_analysis_thread(self):
        self.status_label.config(text="Processing...")
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        try:
            test_path = self.test_file_entry.get()
            datasets = {k: pd.read_csv(v, usecols=['Signal Values', 'Label']) for k, v in DEFAULT_DATASET_FILES.items() if os.path.exists(v)}
            if os.path.exists(test_path):
                datasets['Test'] = pd.read_csv(test_path, usecols=['Signal Values', 'Label'])

            processed = {k: process_dataset(df) for k, df in datasets.items() if df is not None}
            train = pd.concat([processed['Train1'], processed['Train2']], ignore_index=True)
            X_train, y_train = train.drop('Label', axis=1), train['Label']
            X_val, y_val = processed['Validation'].drop('Label', axis=1), processed['Validation']['Label']
            X_test, y_test = processed['Test'].drop('Label', axis=1), processed['Test']['Label']

            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            cm = confusion_matrix(y_test, test_pred)

            self.generate_pdf_report(val_acc, test_acc, cm, datasets['Test'], processed['Test'])
            self.status_label.config(text="Report generated.")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")

    def generate_pdf_report(self, val_acc, test_acc, cm, raw_df, processed_df):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            self.status_label.config(text="Report generation cancelled.")
            return

        doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=1.5*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles for Grid Guardians branding
        styles.add(ParagraphStyle(name='GridGuardiansTitle', fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor(GRID_GUARDIANS_BLUE), spaceAfter=12))
        styles.add(ParagraphStyle(name='GridGuardiansHeading2', fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor(GRID_GUARDIANS_DARK), spaceAfter=8))
        styles.add(ParagraphStyle(name='GridGuardiansHeading3', fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor(GRID_GUARDIANS_DARK), spaceAfter=6))
        styles.add(ParagraphStyle(name='GridGuardiansBody', fontName='Helvetica', fontSize=10, textColor=colors.HexColor(GRID_GUARDIANS_DARK), spaceAfter=6))
        styles.add(ParagraphStyle(name='GridGuardiansFooter', fontName='Helvetica', fontSize=8, textColor=colors.white))

        elements = []

        # Cover page
        if os.path.exists(LOGO_PATH):
            logo = RLImage(LOGO_PATH, width=2*inch, height=2*inch * (Image.open(LOGO_PATH).height / Image.open(LOGO_PATH).width))
            logo.hAlign = 'CENTER'
            elements.append(logo)
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph("Cable Signals Analysis Report", styles['GridGuardiansTitle']))
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph(f"Prepared by Grid Guardians Research Team", styles['GridGuardiansBody']))
        elements.append(Paragraph(f"Date: {date.today()}", styles['GridGuardiansBody']))
        elements.append(PageBreak())

        # Content
        elements.append(Paragraph("Cable Signals Analysis Report", styles['GridGuardiansTitle']))
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph(f"Date: {date.today()}", styles['GridGuardiansBody']))
        elements.append(Spacer(1, 0.25*inch))

        # Summary
        total = len(raw_df)
        pd_count = raw_df['Label'].sum()
        elements.append(Paragraph("1. Summary", styles['GridGuardiansHeading2']))
        elements.append(Paragraph(f"Analyzed {total:,} signals.", styles['GridGuardiansBody']))
        elements.append(Paragraph(f"Detected {pd_count:,} PD signals.", styles['GridGuardiansBody']))
        elements.append(Paragraph(f"Test Accuracy: {test_acc:.2%}", styles['GridGuardiansBody']))
        elements.append(Paragraph(f"Validation Accuracy: {val_acc:.2%}", styles['GridGuardiansBody']))
        elements.append(Spacer(1, 0.1*inch))

        # Recommendations for high PD count
        if pd_count > 1500:
            elements.append(Paragraph("1.1 Recommendations for High PD Count", styles['GridGuardiansHeading3']))
            elements.append(Paragraph(f"PD signal count ({pd_count:,}) exceeds 1500, indicating potential system-wide issues.", styles['GridGuardiansBody']))
            elements.append(Paragraph("Recommended Solutions:", styles['GridGuardiansBody']))
            elements.append(Paragraph("- Enhanced Monitoring: Deploy continuous monitoring systems to track PD activity.", styles['GridGuardiansBody']))
            elements.append(Paragraph("- Cable Replacement: Prioritize replacement of cables with high PD incidence.", styles['GridGuardiansBody']))
            elements.append(Paragraph("- System Diagnostics: Conduct comprehensive diagnostics to identify underlying causes.", styles['GridGuardiansBody']))
            elements.append(Paragraph("- Best Insulation: Use cross-linked polyethylene (XLPE) insulation for superior dielectric strength and resistance to partial discharges.", styles['GridGuardiansBody']))
        else:
            elements.append(Paragraph("1.1 PD Count Status", styles['GridGuardiansHeading3']))
            elements.append(Paragraph(f"PD signal count ({pd_count:,}) is within acceptable limits (<=1500). No immediate action required.", styles['GridGuardiansBody']))
        elements.append(Spacer(1, 0.25*inch))

        # Signal Statistics
        for label, title in [(0, "2.1 Safe Signals"), (1, "2.2 Warning Signals")]:
            desc = processed_df[processed_df['Label'] == label].drop(columns='Label').describe().loc[['min', 'max']]
            elements.append(Paragraph(title, styles['GridGuardiansHeading2']))
            data = [['Feature', 'Min', 'Max']] + [[f, f"{desc.loc['min', f]:.4f}", f"{desc.loc['max', f]:.4f}"] for f in desc.columns]
            table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(GRID_GUARDIANS_DARK)),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(GRID_GUARDIANS_BLUE)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(KeepTogether(table))
            elements.append(Spacer(1, 0.25*inch))

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix", fontsize=12, color=GRID_GUARDIANS_DARK)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
            plt.close(fig)
            cm_image = RLImage(tmp.name, width=4*inch, height=3*inch)
            cm_image.hAlign = 'CENTER'
            elements.append(KeepTogether(cm_image))

        # Hypothesis Test
        pd_std = processed_df[processed_df['Label'] == 1]['std']
        non_pd_std = processed_df[processed_df['Label'] == 0]['std']
        t_stat, p_val = ttest_ind(pd_std, non_pd_std, equal_var=False)
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph("3. Hypothesis Test (95% Confidence Level)", styles['GridGuardiansHeading2']))
        elements.append(Paragraph(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4e}", styles['GridGuardiansBody']))
        elements.append(Paragraph(f"Result: {'Statistically significant' if p_val < 0.05 else 'Not statistically significant'} (α = 0.05).", styles['GridGuardiansBody']))
        elements.append(Spacer(1, 0.25*inch))

        # Separate PD and Non-PD Plots
        try:
            pd_signal = np.fromstring(raw_df[raw_df['Label'] == 1]['Signal Values'].iloc[0], sep=',')
            non_pd_signal = np.fromstring(raw_df[raw_df['Label'] == 0]['Signal Values'].iloc[0], sep=',')

            # PD Signal Plot
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(pd_signal[:200], label='PD', color='red')
            ax.legend()
            ax.set_title('Sample PD Signal', fontsize=12, color=GRID_GUARDIANS_DARK)
            ax.grid(True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
                plt.close(fig)
                pd_image = RLImage(tmp.name, width=5*inch, height=2.5*inch)
                pd_image.hAlign = 'CENTER'
                elements.append(KeepTogether(pd_image))
            elements.append(Spacer(1, 0.1*inch))

            # Non-PD Signal Plot
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(non_pd_signal[:200], label='Non-PD', color='green')
            ax.legend()
            ax.set_title('Sample Non-PD Signal', fontsize=12, color=GRID_GUARDIANS_DARK)
            ax.grid(True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
                plt.close(fig)
                non_pd_image = RLImage(tmp.name, width=5*inch, height=2.5*inch)
                non_pd_image.hAlign = 'CENTER'
                elements.append(KeepTogether(non_pd_image))
        except Exception as e:
            elements.append(Paragraph(f"Plot Error: {e}", styles['GridGuardiansBody']))

        # Custom header and footer
        def add_header_footer(canvas, doc):
            canvas.saveState()
            
            # Header
            canvas.setFillColor(colors.HexColor(GRID_GUARDIANS_BLUE))
            canvas.rect(0, letter[1] - 1*inch, letter[0], 1*inch, fill=True)
            if os.path.exists(LOGO_PATH):
                canvas.drawImage(LOGO_PATH, 0.5*inch, letter[1] - 0.9*inch, width=0.75*inch, height=0.75*inch * (Image.open(LOGO_PATH).height / Image.open(LOGO_PATH).width), mask='auto')
            canvas.setFont('Helvetica-Bold', 12)
            canvas.setFillColor(colors.white)
            canvas.drawString(1.5*inch, letter[1] - 0.65*inch, "Cable Signals Analysis Report")
            
            # Footer
            canvas.setFillColor(colors.HexColor(GRID_GUARDIANS_BLUE))
            canvas.rect(0, 0, letter[0], 0.5*inch, fill=True)
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.white)
            canvas.drawString(0.5*inch, 0.2*inch, f"© {date.today().year} Grid Guardians | Research Day")
            canvas.drawRightString(letter[0] - 0.5*inch, 0.2*inch, f"Page {doc.page}")
            
            canvas.restoreState()

        doc.build(elements, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

if _name_ == '_main_':
    root = tk.Tk()
    app = SignalsAnalysisApp(root)
    root.mainloop()
