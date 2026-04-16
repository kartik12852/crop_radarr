"""
utils/pdf_exporter.py
PDF climate risk reports using fpdf2.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from io import BytesIO
import pandas as pd
import numpy as np

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from src.model_utils import LABEL_NAMES, LABEL_COLORS

SEV_COLORS = {"Low": (46,204,113), "Medium": (241,196,15),
              "High": (231,76,60),  "Critical": (192,57,43)}
SEV_MAP    = {0: "Low", 1: "High", 2: "High", 3: "Critical", 4: "Medium"}


class ClimateReportPDF:
    def __init__(self):
        if not FPDF_AVAILABLE:
            raise ImportError("fpdf2 not installed. Run: pip install fpdf2")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def _header(self, title: str, subtitle: str = ""):
        p = self.pdf
        p.add_page()
        p.set_fill_color(34, 139, 34)
        p.rect(0, 0, 210, 32, "F")
        p.set_text_color(255, 255, 255)
        p.set_font("Helvetica", "B", 18)
        p.cell(0, 12, "Climate Radar - Crop Risk Intelligence", ln=True)
        p.set_font("Helvetica", "B", 13)
        p.cell(0, 8, title, ln=True)
        if subtitle:
            p.set_font("Helvetica", "", 10)
            p.cell(0, 6, subtitle, ln=True)
        p.set_text_color(0, 0, 0)
        p.ln(8)

    def _section(self, heading: str):
        p = self.pdf
        p.set_font("Helvetica", "B", 12)
        p.set_fill_color(230, 245, 230)
        p.cell(0, 8, f"  {heading}", ln=True, fill=True)
        p.ln(2)

    def _bullet(self, text: str, indent: int = 10):
        p = self.pdf
        p.set_font("Helvetica", "", 10)
        p.set_x(indent)
        p.multi_cell(0, 6, f"- {text}")

    def _kv(self, key: str, value: str):
        p = self.pdf
        p.set_font("Helvetica", "B", 10)
        p.set_x(10)
        p.cell(55, 6, key + ":", ln=False)
        p.set_font("Helvetica", "", 10)
        p.multi_cell(0, 6, str(value))

    def generate_report(self, zones_df, preds, probs, metrics,
                        recommendations=None) -> bytes:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._header("Crop Climate Risk Assessment Report",
                     f"Generated: {ts} | {len(zones_df)} Agricultural Zones")

        # Executive Summary
        self._section("Executive Summary")
        risk_counts = pd.Series(preds).value_counts().sort_index()
        for rid, cnt in risk_counts.items():
            self._bullet(f"{LABEL_NAMES.get(int(rid), str(rid))}: "
                         f"{cnt} zones ({cnt/len(preds)*100:.1f}%)")
        self.pdf.ln(4)

        # Metrics
        if metrics:
            self._section("Model Performance Metrics")
            for k, v in metrics.items():
                self._kv(k.replace("_", " ").title(), str(v))
            self.pdf.ln(4)

        # Zone table
        self._section("Zone Risk Summary")
        p = self.pdf
        p.set_font("Helvetica", "B", 9)
        col_w   = [55, 28, 28, 35, 25, 20]
        headers = ["Zone Name", "Crop", "Risk", "Risk Name", "Confidence%", "Severity"]
        for i, h in enumerate(headers):
            p.cell(col_w[i], 7, h, border=1, fill=True)
        p.ln()

        for idx, (_, row) in enumerate(zones_df.iterrows()):
            pred  = int(preds[idx])
            rname = LABEL_NAMES.get(pred, "?")
            conf  = f"{probs[idx].max() * 100:.1f}"
            sev   = SEV_MAP.get(pred, "Low")
            c     = SEV_COLORS.get(sev, (200, 200, 200))
            p.set_font("Helvetica", "", 9)
            p.cell(col_w[0], 6, str(row.get("zone_name", ""))[:28], border=1)
            p.cell(col_w[1], 6, str(row.get("crop", ""))[:14],      border=1)
            p.cell(col_w[2], 6, str(pred),                          border=1)
            p.set_fill_color(*c)
            p.cell(col_w[3], 6, rname, border=1, fill=True)
            p.set_fill_color(255, 255, 255)
            p.cell(col_w[4], 6, conf,  border=1)
            p.cell(col_w[5], 6, sev,   border=1)
            p.ln()

        # Recommendations
        if recommendations:
            self.pdf.add_page()
            self._header("Zone-wise Recommendations", f"Generated: {ts}")
            for rec in recommendations[:10]:
                self._section(f"{rec['zone_name']} - {rec['crop']} - {rec['risk_name']}")
                for act in rec.get("immediate", [])[:3]:
                    self._bullet(act, indent=14)
                if rec.get("crop_tip"):
                    self._kv("Crop-specific Tip", rec["crop_tip"])
                self.pdf.ln(2)

        # Footer
        p = self.pdf
        p.set_y(-18)
        p.set_font("Helvetica", "I", 8)
        p.set_text_color(120, 120, 120)
        p.cell(0, 6, "Climate Radar System | Predictions are advisory only.", align="C")

        pdf_bytes = self.pdf.output()
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1")
        return bytes(pdf_bytes)


def generate_pdf_report(zones_df, preds, probs, metrics, recommendations=None) -> bytes:
    return ClimateReportPDF().generate_report(zones_df, preds, probs, metrics, recommendations)
