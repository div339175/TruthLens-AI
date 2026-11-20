"""
TruthLens AI — PowerPoint Presentation Generator
Creates an authoritative, 10-slide academic research presentation
matching the exact design specifications, color palette, and content requirements.
"""

import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- COLOR PALETTE SPECIFICATIONS ---
COLOR_BG_DARK = RGBColor(7, 13, 30)       # #070D1E Deep Navy
COLOR_BG_CENTER = RGBColor(15, 29, 56)    # #0F1D38 Dark Slate
COLOR_CARD_BG = RGBColor(19, 34, 71)      # #132247 Glassmorphic Navy Card
COLOR_CARD_BORDER = RGBColor(30, 58, 138) # #1E3A8A Dark Blue Border
COLOR_CYAN = RGBColor(0, 210, 255)        # #00D2FF Electric Cyan Accent
COLOR_BLUE = RGBColor(58, 134, 255)       # #3A86FF Cobalt Blue Secondary
COLOR_WHITE = RGBColor(255, 255, 255)     # #FFFFFF Crisp White
COLOR_SLATE = RGBColor(148, 163, 184)     # #94A3B8 Slate Gray
COLOR_GREEN = RGBColor(16, 185, 129)      # #10B981 Emerald Green (Real/Success)
COLOR_RED = RGBColor(239, 68, 68)         # #EF4444 Crimson Red (Fake/Defect)
COLOR_AMBER = RGBColor(245, 158, 11)      # #F59E0B Amber (Uncertain/Warning)
COLOR_DARK_ROW = RGBColor(12, 23, 49)     # Alternating table row dark
COLOR_HEADER_ROW = RGBColor(25, 48, 98)   # Table header row

def set_slide_background(slide):
    """Adds a dark navy base background rectangle to the slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_DARK
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category="CSET 346 · NATURAL LANGUAGE PROCESSING · TRUTHLENS AI"):
    """Adds a standardized top header banner with category badge and crisp white title."""
    # Category badge
    cat_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(4.5), Inches(0.32))
    cat_box.fill.solid()
    cat_box.fill.fore_color.rgb = COLOR_CARD_BG
    cat_box.line.color.rgb = COLOR_CYAN
    cat_box.line.width = Pt(1)
    tf = cat_box.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = category
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.alignment = PP_ALIGN.CENTER

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.6))
    tf2 = title_box.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = title_text
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_WHITE

def create_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER, border_width=1):
    """Creates a glassmorphic rounded card container."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(border_width)
    else:
        card.line.fill.background()
    return card

def add_styled_table(slide, left, top, width, height, headers, rows_data, col_widths, highlight_row_idx=-1):
    """Adds a high-contrast styled data table to the slide."""
    table_shape = slide.shapes.add_table(len(rows_data) + 1, len(headers), left, top, width, height)
    table = table_shape.table

    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    # Format Header Row
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_HEADER_ROW
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = header
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_CYAN
        p.alignment = PP_ALIGN.CENTER

    # Format Data Rows
    for row_idx, row in enumerate(rows_data):
        is_highlight = (row_idx == highlight_row_idx)
        bg = COLOR_CARD_BG if (row_idx % 2 == 0) else COLOR_DARK_ROW
        if is_highlight:
            bg = RGBColor(22, 50, 95)

        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(9.5)
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = COLOR_CYAN if col_idx in [0, len(row)-1] else COLOR_WHITE
            else:
                p.font.color.rgb = COLOR_WHITE if col_idx == 0 else COLOR_SLATE
            p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    return table_shape


def build_slide_1(prs):
    """SLIDE 1: Title & Team Identity"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)

    # Top Category Badge
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.666), Inches(0.6), Inches(6.0), Inches(0.38))
    badge.fill.solid()
    badge.fill.fore_color.rgb = COLOR_CARD_BG
    badge.line.color.rgb = COLOR_CYAN
    badge.line.width = Pt(1.5)
    tf = badge.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = "MILESTONE 1 · CSET 346 · NATURAL LANGUAGE PROCESSING"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.alignment = PP_ALIGN.CENTER

    # Main Hero Title
    title_box = slide.shapes.add_textbox(Inches(1.5), Inches(1.15), Inches(10.333), Inches(0.9))
    tf2 = title_box.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = "TruthLens AI"
    p2.font.size = Pt(44)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_WHITE
    p2.alignment = PP_ALIGN.CENTER

    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(1.5), Inches(2.05), Inches(10.333), Inches(0.5))
    tf3 = sub_box.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "Multimodal NLP for Deepfake Detection and Explainable Content Verification"
    p3.font.size = Pt(17)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_CYAN
    p3.alignment = PP_ALIGN.CENTER

    # Author Card 1 (Left Column)
    card1 = create_card(slide, Inches(1.0), Inches(2.75), Inches(5.45), Inches(2.2), border_color=COLOR_BLUE)
    tf_c1 = card1.text_frame
    tf_c1.word_wrap = True
    tf_c1.vertical_anchor = MSO_ANCHOR.TOP
    p_c1_title = tf_c1.paragraphs[0]
    p_c1_title.text = "RESEARCH TEAM · PRIMARY AUTHORS"
    p_c1_title.font.size = Pt(11)
    p_c1_title.font.bold = True
    p_c1_title.font.color.rgb = COLOR_CYAN
    
    authors_left = [
        ("Divakar Maurya", "S24CSEU0807", "s24cseu0807@bennett.edu.in"),
        ("Vrishank Pathak", "S24CSEU0950", "s24cseu0950@bennett.edu.in"),
        ("Hrishit Goel", "S24CSEU2861", "s24cseu2861@bennett.edu.in")
    ]
    for name, roll, email in authors_left:
        p = tf_c1.add_paragraph()
        p.text = f"• {name}  [{roll}] — {email}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(4)

    # Author Card 2 (Right Column)
    card2 = create_card(slide, Inches(6.883), Inches(2.75), Inches(5.45), Inches(2.2), border_color=COLOR_BLUE)
    tf_c2 = card2.text_frame
    tf_c2.word_wrap = True
    tf_c2.vertical_anchor = MSO_ANCHOR.TOP
    p_c2_title = tf_c2.paragraphs[0]
    p_c2_title.text = "COLLABORATORS & INSTITUTIONAL AFFILIATION"
    p_c2_title.font.size = Pt(11)
    p_c2_title.font.bold = True
    p_c2_title.font.color.rgb = COLOR_CYAN

    authors_right = [
        ("Dhruv Agarwal", "S24CSEU0916", "s24cseu0916@bennett.edu.in"),
        ("Vishwas Bhargava", "S24CSEU0190", "s24cseu0190@bennett.edu.in")
    ]
    for name, roll, email in authors_right:
        p = tf_c2.add_paragraph()
        p.text = f"• {name}  [{roll}] — {email}"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(4)
    
    p_aff = tf_c2.add_paragraph()
    p_aff.text = "Dept. of Computer Science & Engineering\nBennett University, Greater Noida, India"
    p_aff.font.size = Pt(9.5)
    p_aff.font.bold = True
    p_aff.font.color.rgb = COLOR_SLATE
    p_aff.space_before = Pt(6)

    # Bottom Stream Convergence Diagram
    flow_bg = create_card(slide, Inches(1.0), Inches(5.2), Inches(11.333), Inches(1.7), border_color=COLOR_CYAN, border_width=1.5)
    tf_flow = flow_bg.text_frame
    tf_flow.word_wrap = True
    p_f_title = tf_flow.paragraphs[0]
    p_f_title.text = "UNIFIED MULTIMODAL VERIFICATION FLOW"
    p_f_title.font.size = Pt(11)
    p_f_title.font.bold = True
    p_f_title.font.color.rgb = COLOR_CYAN
    p_f_title.alignment = PP_ALIGN.CENTER

    streams = [
        ("Text Stream", "DistilBERT L=128", Inches(1.2), Inches(5.7)),
        ("Image Stream", "ResNet18 224x224", Inches(3.2), Inches(5.7)),
        ("Video Stream", "8-Frame Latent Pool", Inches(5.2), Inches(5.7)),
        ("Audio Stream", "128-D STFT Spectrum", Inches(7.2), Inches(5.7))
    ]
    for name, sub, x, y in streams:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(1.8), Inches(0.9))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(12, 23, 49)
        box.line.color.rgb = COLOR_BLUE
        tf_b = box.text_frame
        tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
        p1 = tf_b.paragraphs[0]
        p1.text = name
        p1.font.bold = True
        p1.font.size = Pt(10)
        p1.font.color.rgb = COLOR_WHITE
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf_b.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = COLOR_SLATE
        p2.alignment = PP_ALIGN.CENTER

    # Convergence Box (Core + Verdict)
    core_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.3), Inches(5.7), Inches(2.8), Inches(0.9))
    core_box.fill.solid()
    core_box.fill.fore_color.rgb = RGBColor(22, 50, 95)
    core_box.line.color.rgb = COLOR_CYAN
    core_box.line.width = Pt(1.5)
    tf_core = core_box.text_frame
    tf_core.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_c1 = tf_core.paragraphs[0]
    p_c1.text = "TruthLens Multimodal Core"
    p_c1.font.bold = True
    p_c1.font.size = Pt(10.5)
    p_c1.font.color.rgb = COLOR_CYAN
    p_c1.alignment = PP_ALIGN.CENTER
    p_c2 = tf_core.add_paragraph()
    p_c2.text = "REAL · FAKE · UNCERTAIN (τ = 0.65)"
    p_c2.font.bold = True
    p_c2.font.size = Pt(8.5)
    p_c2.font.color.rgb = COLOR_GREEN
    p_c2.alignment = PP_ALIGN.CENTER


def build_slide_2(prs):
    """SLIDE 2: The Problem & Literature Gap: Deepfakes Are Multimodal"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "The Problem & Literature Gap: Deepfakes Are Multimodal", "01 · RESEARCH CONTEXT & MOTIVATION")

    # Left Column: 3 Problem Cards (40% width = 4.6 inches)
    problems = [
        ("The Cheapfake Paradox", "Real, unmanipulated photographs paired with fabricated, emotionally charged headlines bypass computer vision filters entirely.", COLOR_AMBER),
        ("Visual Hegemony", "High-capacity vision backbones overpower text signals during naive late fusion, causing composite fakes to be misclassified as authentic.", COLOR_RED),
        ("Overconfident Black Boxes", "Unimodal classifiers frequently output >0.99 confidence without admitted uncertainty or explainable, causal evidence for human review.", COLOR_CYAN)
    ]
    top_pos = Inches(1.5)
    for title, desc, accent in problems:
        p_card = create_card(slide, Inches(0.8), top_pos, Inches(4.5), Inches(1.4), border_color=accent)
        tf = p_card.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        p_t = tf.paragraphs[0]
        p_t.text = f"⚠ {title}"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = accent
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = COLOR_WHITE
        p_d.space_before = Pt(4)
        top_pos += Inches(1.52)

    # Right Column: Styled Literature Table (60% width = 6.9 inches)
    headers = ["Work & Ref.", "Modality", "Core Mechanism", "Critical Limitation"]
    rows = [
        ["Rössler (FaceForensics++)", "Vision", "CNN pixel artifacts", "Blind to unmanipulated image reuse"],
        ["Guera & Delp (AVSS 2018)", "Video", "Frame temporal CNN", "Lacked multimodal text context"],
        ["Aneja et al. (COSMOS 2021)", "Text+Image", "Contrastive cheapfake", "High false alarms; uncalibrated"],
        ["Nakamura (Fakeddit 2020)", "Text+Image", "Late / Concat Fusion", "Severe collapse on informal slang"],
        ["Li et al. (IEEE TASLP 2021)", "Audio", "STFT vocoder cues", "Unimodal speech without video sync"],
        ["TruthLens AI (This Work)", "T+I+V+A", "Bidirectional Cross-Attn", "Unifies 4 streams, XAI, τ=0.65"]
    ]
    col_widths = [Inches(1.8), Inches(0.9), Inches(1.9), Inches(2.4)]
    add_styled_table(slide, Inches(5.6), Inches(1.5), Inches(7.0), Inches(4.3), headers, rows, col_widths, highlight_row_idx=5)

    # Bottom Full-Width Ribbon
    ribbon = create_card(slide, Inches(0.8), Inches(6.1), Inches(11.8), Inches(0.85), bg_color=RGBColor(15, 29, 56), border_color=COLOR_CYAN)
    tf_r = ribbon.text_frame
    tf_r.word_wrap = True
    tf_r.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_r1 = tf_r.paragraphs[0]
    p_r1.text = "RESEARCH GAP & TRUTHLENS SOLUTION"
    p_r1.font.bold = True
    p_r1.font.size = Pt(10)
    p_r1.font.color.rgb = COLOR_CYAN
    p_r2 = tf_r.add_paragraph()
    p_r2.text = "Current systems operate in unimodal silos and force speculative binary labels. TruthLens AI unifies text, image, multi-frame video, and acoustic spectrum with calibrated uncertainty routing (τ = 0.65) and transparent multimodal explainability."
    p_r2.font.size = Pt(9)
    p_r2.font.color.rgb = COLOR_WHITE


def build_slide_3(prs):
    """SLIDE 3: Audited Datasets & Multi-Suite Setup"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Audited Datasets & Multi-Suite Evaluation Setup", "02 · EXPERIMENTAL BENCHMARKS")

    col_w = Inches(3.75)
    gap = Inches(0.28)
    top_pos = Inches(1.5)
    h_card = Inches(5.4)

    # Column 1: Primary Course Dataset
    c1 = create_card(slide, Inches(0.8), top_pos, col_w, h_card, border_color=COLOR_CYAN)
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "PRIMARY COURSE DATASET"
    p1.font.bold = True
    p1.font.size = Pt(12)
    p1.font.color.rgb = COLOR_CYAN

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "N = 320 Balanced Multimodal Instances"
    p1_sub.font.bold = True
    p1_sub.font.size = Pt(10.5)
    p1_sub.font.color.rgb = COLOR_WHITE
    p1_sub.space_before = Pt(2)

    # Split Meter inside Column 1
    p_split = tf1.add_paragraph()
    p_split.text = "• 70% Train (N = 224) — Clean class balance\n• 15% Validation (N = 48) — Model checkpointing\n• 15% Test (N = 48) — Zero visual/textual leakage"
    p_split.font.size = Pt(9.5)
    p_split.font.color.rgb = COLOR_SLATE
    p_split.space_before = Pt(8)

    p_typ = tf1.add_paragraph()
    p_typ.text = "Four Forensic Typologies (40 pairs each):"
    p_typ.font.bold = True
    p_typ.font.size = Pt(10)
    p_typ.font.color.rgb = COLOR_BLUE
    p_typ.space_before = Pt(10)

    typologies = [
        "1. Facial Reenactment & Deepfakes",
        "2. Latent Diffusion Image Synthesis",
        "3. Spliced Headlines & Mismatched Captions",
        "4. Cloned & Synthesized Speech Audio"
    ]
    for t in typologies:
        p = tf1.add_paragraph()
        p.text = f"  - {t}"
        p.font.size = Pt(9)
        p.font.color.rgb = COLOR_WHITE

    # Column 2: External Generalization Suites
    c2 = create_card(slide, Inches(0.8) + col_w + gap, top_pos, col_w, h_card, border_color=COLOR_BLUE)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "EXTERNAL BENCHMARK SUITES"
    p2.font.bold = True
    p2.font.size = Pt(12)
    p2.font.color.rgb = COLOR_BLUE

    p2_sub = tf2.add_paragraph()
    p2_sub.text = "N = 180 Out-of-Domain Generalization"
    p2_sub.font.bold = True
    p2_sub.font.size = Pt(10.5)
    p2_sub.font.color.rgb = COLOR_WHITE
    p2_sub.space_before = Pt(2)

    benchmarks = [
        ("Fakeddit Social Benchmark", "N = 60 Pairs (30 Real / 30 Fake)", "Uncurated Reddit posts with slang, heavy irony, and noisy multimodal contexts."),
        ("Deepfake Audio v4", "N = 60 Audio Clips (30 Real / 30 Fake)", "Human natural speech vs. ElevenLabs and Hume AI generative voice clones."),
        ("Generative Video Sora", "N = 60 Video Clips (30 Real / 30 Fake)", "Documentary sequences vs. OpenAI Sora & Google Veo synthetic video generations.")
    ]
    for b_title, b_stat, b_desc in benchmarks:
        p_bt = tf2.add_paragraph()
        p_bt.text = f"• {b_title}"
        p_bt.font.bold = True
        p_bt.font.size = Pt(10)
        p_bt.font.color.rgb = COLOR_CYAN
        p_bt.space_before = Pt(6)

        p_bs = tf2.add_paragraph()
        p_bs.text = f"  {b_stat} — {b_desc}"
        p_bs.font.size = Pt(8.8)
        p_bs.font.color.rgb = COLOR_SLATE

    p_disc = tf2.add_paragraph()
    p_disc.text = "Transparency Note: FakeAVCeleb is institutional gated data; rigorously surrogated via Audio v4 & Sora suites."
    p_disc.font.size = Pt(8.5)
    p_disc.font.color.rgb = COLOR_AMBER
    p_disc.space_before = Pt(8)

    # Column 3: Controlled Semantic Mismatch Suite
    c3 = create_card(slide, Inches(0.8) + (col_w + gap)*2, top_pos, col_w, h_card, border_color=COLOR_AMBER)
    tf3 = c3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "SEMANTIC MISMATCH SUITE"
    p3.font.bold = True
    p3.font.size = Pt(12)
    p3.font.color.rgb = COLOR_AMBER

    p3_sub = tf3.add_paragraph()
    p3_sub.text = "30 Pairs / 60 Controlled Evaluations"
    p3_sub.font.bold = True
    p3_sub.font.size = Pt(10.5)
    p3_sub.font.color.rgb = COLOR_WHITE
    p3_sub.space_before = Pt(2)

    p3_desc = tf3.add_paragraph()
    p3_desc.text = "Isolates cross-modal conflict detection across four strictly controlled conditions:"
    p3_desc.font.size = Pt(9.2)
    p3_desc.font.color.rgb = COLOR_SLATE
    p3_desc.space_before = Pt(6)

    conditions = [
        ("Condition 1: Authentic Baseline", "Real Text + Real Matching Image", COLOR_GREEN),
        ("Condition 2: The Cheapfake Conflict", "Fabricated Text + Real Image (Visual Dominance Trap)", COLOR_RED),
        ("Condition 3: Contextual Swap", "Real Text + Spliced Out-of-Context Media", COLOR_AMBER),
        ("Condition 4: Cross-Modal Synthetic", "Fabricated Text + Generative AI Media", COLOR_BLUE)
    ]
    for cond_title, cond_desc, col_acc in conditions:
        p_ct = tf3.add_paragraph()
        p_ct.text = f"• {cond_title}"
        p_ct.font.bold = True
        p_ct.font.size = Pt(9.5)
        p_ct.font.color.rgb = col_acc
        p_ct.space_before = Pt(4)

        p_cd = tf3.add_paragraph()
        p_cd.text = f"  {cond_desc}"
        p_cd.font.size = Pt(8.6)
        p_cd.font.color.rgb = COLOR_WHITE


def build_slide_4(prs):
    """SLIDE 4: Overall System Architecture"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "End-to-End Multimodal Deepfake Verification Flow", "03 · SYSTEM ARCHITECTURE")

    # 4 Sequential Horizontal Stages
    stage_w = Inches(2.75)
    gap = Inches(0.24)
    top_pos = Inches(1.5)
    h_stage = Inches(4.5)

    stages = [
        ("STAGE 1: INPUTS", COLOR_SLATE, [
            ("Text Claim T", "News headlines, social media posts, captions"),
            ("Static Image I", "224x224 RGB image (JPEG/PNG)"),
            ("Video Stream V", "8-frame uniform temporal extraction"),
            ("Audio Waveform A", "16 kHz mono waveform (WAV/MP3)")
        ]),
        ("STAGE 2: ENCODERS", COLOR_BLUE, [
            ("DistilBERT Text", "WordPiece (L=128) -> h_[CLS] in R^768 -> z_T in R^256"),
            ("ResNet18 Visual", "Conv Layer4 -> AvgPool -> z_I in R^256"),
            ("Temporal Video", "8-frame latent feature mean pool -> z_V in R^256"),
            ("STFT Acoustic", "128-D spectral vector (Mel+ZCR+HF) -> z_A in R^256")
        ]),
        ("STAGE 3: FUSION", COLOR_CYAN, [
            ("Cross-Attention", "4-head bidirectional text <-> vision attention"),
            ("Contrastive InfoNCE", "Pulls semantic matches, penalizes cheapfakes (λ=0.05)"),
            ("Projection Layer", "Concat [z_T, z_V, z_A] in R^768 -> R^256"),
            ("Classification Head", "Linear + Softmax -> P(Fake), P(Real)")
        ]),
        ("STAGE 4: VERDICT & XAI", COLOR_GREEN, [
            ("Temperature Scaling", "T* = 1.4628 softens overconfident predictions"),
            ("Uncertainty Routing", "Threshold τ = 0.65 -> REAL / FAKE / UNCERTAIN"),
            ("Explainable XAI", "Layer4 Grad-CAM heatmaps + Token saliency"),
            ("Persistent Audit", "SQLite logging (truthlens.db) for compliance")
        ])
    ]

    for idx, (title, color_b, items) in enumerate(stages):
        left_pos = Inches(0.8) + idx * (stage_w + gap)
        card = create_card(slide, left_pos, top_pos, stage_w, h_stage, border_color=color_b, border_width=1.5)
        tf = card.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(11)
        p_t.font.color.rgb = color_b
        p_t.alignment = PP_ALIGN.CENTER

        for item_title, item_desc in items:
            p_it = tf.add_paragraph()
            p_it.text = f"• {item_title}"
            p_it.font.bold = True
            p_it.font.size = Pt(9.5)
            p_it.font.color.rgb = COLOR_WHITE
            p_it.space_before = Pt(6)

            p_id = tf.add_paragraph()
            p_id.text = item_desc
            p_id.font.size = Pt(8.5)
            p_id.font.color.rgb = COLOR_SLATE

    # Bottom Pipeline Summary
    bot_box = create_card(slide, Inches(0.8), Inches(6.15), Inches(11.75), Inches(0.85), bg_color=COLOR_CARD_BG, border_color=COLOR_CYAN)
    tf_b = bot_box.text_frame
    tf_b.word_wrap = True
    tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_bt = tf_b.paragraphs[0]
    p_bt.text = "KEY ARCHITECTURAL HIGHLIGHT"
    p_bt.font.bold = True
    p_bt.font.size = Pt(10)
    p_bt.font.color.rgb = COLOR_CYAN
    p_bd = tf_b.add_paragraph()
    p_bd.text = "Features pass through independent modality projectors into a unified R^256 latent space before bidirectional cross-attention fusion, preventing high-capacity vision features from dominating linguistic claims."
    p_bd.font.size = Pt(9)
    p_bd.font.color.rgb = COLOR_WHITE


def build_slide_5(prs):
    """SLIDE 5: Multimodal Methodology & Temporal Video Fix"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Multimodal Encoders & Critical Video Temporal Fix", "04 · METHODOLOGY & ARCHITECTURAL REFINEMENTS")

    w_card = Inches(5.75)
    h_card = Inches(2.55)
    gap_x = Inches(0.3)
    gap_y = Inches(0.25)
    top_row = Inches(1.5)
    bot_row = top_row + h_card + gap_y

    # Top-Left: DistilBERT Text
    c_tl = create_card(slide, Inches(0.8), top_row, w_card, h_card, border_color=COLOR_BLUE)
    tf_tl = c_tl.text_frame
    tf_tl.word_wrap = True
    p = tf_tl.paragraphs[0]
    p.text = "1. TEXT BRANCH — DistilBERT & Domain Adaptation"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_CYAN

    points_tl = [
        "WordPiece Tokenizer (max sequence L = 128) -> DistilBERT sequence H_T in R^{128 x 768}",
        "Latent Projection: z_T = LayerNorm(W_T h_[CLS] + b_T) in R^256 with GELU activation",
        "Rehearsal Regularization: L_adapt = L_CE(T_fakeddit) + 0.5 * L_CE(T_course)",
        "Impact: Increases informal social media accuracy by +23.34% while preventing catastrophic forgetting."
    ]
    for pt in points_tl:
        p = tf_tl.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(8.8)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(3)

    # Top-Right: ResNet18 Image
    c_tr = create_card(slide, Inches(0.8) + w_card + gap_x, top_row, w_card, h_card, border_color=COLOR_BLUE)
    tf_tr = c_tr.text_frame
    tf_tr.word_wrap = True
    p = tf_tr.paragraphs[0]
    p.text = "2. IMAGE BRANCH — ResNet18 Convolutional Backbone"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_CYAN

    points_tr = [
        "Standardized Preprocessing: 224x224 RGB inputs normalized with ImageNet mean and variance",
        "Spatial Feature Maps: Layer4 residual feature tensor H_V in R^{512 x 7 x 7}",
        "Global Projection: z_I = LayerNorm(W_I AvgPool(H_V) + b_I) in R^256",
        "Explainability Hook: Retains Layer4 activation maps for post-hoc Grad-CAM saliency localization."
    ]
    for pt in points_tr:
        p = tf_tr.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(8.8)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(3)

    # Bottom-Left: Video Temporal Pooling (ARCHITECTURAL FIX)
    c_bl = create_card(slide, Inches(0.8), bot_row, w_card, h_card, border_color=COLOR_AMBER, border_width=1.5)
    tf_bl = c_bl.text_frame
    tf_bl.word_wrap = True
    p = tf_bl.paragraphs[0]
    p.text = "3. VIDEO BRANCH — Feature-Level Temporal Pooling [CRITICAL FIX]"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_AMBER

    points_bl = [
        "Phase 2 Defect Resolved: Early prototypes computed pixel mean across frames before CNN, washing out facial motion boundaries and causing massive blur.",
        "Engineered Solution: 8 uniform frames -> 8 independent ResNet forward passes -> Latent Mean Pool in feature space:",
        "    f_video = (1/K) * sum_{k=1}^K f_k in R^512 -> z_V in R^256",
        "Verification: Eliminates pixel smearing; achieves Video ROC-AUC = 0.8200 on generative Sora benchmark."
    ]
    for pt in points_bl:
        p = tf_bl.add_paragraph()
        p.text = pt
        p.font.size = Pt(8.8)
        p.font.color.rgb = COLOR_WHITE if "Engineered" not in pt else COLOR_GREEN
        p.space_before = Pt(3)

    # Bottom-Right: Audio STFT Projections
    c_br = create_card(slide, Inches(0.8) + w_card + gap_x, bot_row, w_card, h_card, border_color=COLOR_CYAN)
    tf_br = c_br.text_frame
    tf_br.word_wrap = True
    p = tf_br.paragraphs[0]
    p.text = "4. AUDIO BRANCH — STFT Spectral Feature Pipeline"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_CYAN

    points_br = [
        "16 kHz Mono Audio Waveforms converted via Short-Time Fourier Transform (STFT)",
        "128-D Acoustic Diagnostics Vector: 120 log-mel frequency bands + Zero Crossing Rate + High-Frequency Spectral Ratio (>4 kHz)",
        "Trainable Linear Projection: z_A = LayerNorm(W_A a + b_A) in R^256",
        "Rigorous Backprop Verification: Unit tests verify non-zero gradients throughout all STFT projection weights (Audio ROC-AUC = 0.8100)."
    ]
    for pt in points_br:
        p = tf_br.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(8.8)
        p.font.color.rgb = COLOR_WHITE
        p.space_before = Pt(3)


def build_slide_6(prs):
    """SLIDE 6: From Prediction to Evidence: Attention & XAI"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "From Prediction to Evidence: Cross-Attention & Multimodal XAI", "05 · CROSS-MODAL FUSION & EXPLAINABILITY")

    w_col = Inches(5.75)
    h_col = Inches(5.4)
    gap = Inches(0.3)
    top_pos = Inches(1.5)

    # Left Column: Bidirectional Cross-Attention Mechanics
    c_left = create_card(slide, Inches(0.8), top_pos, w_col, h_col, border_color=COLOR_CYAN)
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "BIDIRECTIONAL CROSS-ATTENTION FUSION"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_CYAN

    points_l = [
        ("Multi-Head Attention (h = 4, d_k = 64):", "Enables linguistic tokens to query visual regions and visual features to query text tokens simultaneously."),
        ("Text-to-Vision Attention:", "Attn(Q_T, K_V, V_V) = softmax((Q_T K_V^T) / sqrt(d_k)) V_V"),
        ("Vision-to-Text Attention:", "Attn(Q_V, K_T, V_T) = softmax((Q_V K_T^T) / sqrt(d_k)) V_T"),
        ("Cross-Modal Semantic Residuals:", "M_T = LayerNorm(z_T + Dropout(Attn_{T->V}))\nM_V = LayerNorm(z_V + Dropout(Attn_{V->T}))"),
        ("Contrastive Alignment Loss (InfoNCE, λ = 0.05):", "Maximizes cosine similarity between matching text-image pairs while repelling fabricated mismatches in latent space:")
    ]
    for pt_t, pt_d in points_l:
        p1 = tf_l.add_paragraph()
        p1.text = f"• {pt_t}"
        p1.font.bold = True
        p1.font.size = Pt(9.5)
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(5)

        p2 = tf_l.add_paragraph()
        p2.text = f"  {pt_d}"
        p2.font.size = Pt(8.8)
        p2.font.color.rgb = COLOR_SLATE

    # Formula Callout Box
    f_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(5.7), Inches(5.35), Inches(0.95))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = RGBColor(12, 23, 49)
    f_box.line.color.rgb = COLOR_BLUE
    tf_f = f_box.text_frame
    tf_f.vertical_anchor = MSO_ANCHOR.MIDDLE
    pf = tf_f.paragraphs[0]
    pf.text = "L_total = L_CE + 0.05 * L_contrastive\nIncreases semantic mismatch detection from 0.0% to 100.0%!"
    pf.font.bold = True
    pf.font.size = Pt(9.5)
    pf.font.color.rgb = COLOR_GREEN
    pf.alignment = PP_ALIGN.CENTER

    # Right Column: Multimodal Causal Explainability Engine
    c_right = create_card(slide, Inches(0.8) + w_col + gap, top_pos, w_col, h_col, border_color=COLOR_BLUE)
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "CAUSAL EXPLAINABILITY ENGINE"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_BLUE

    xai_cards = [
        ("1. Text Token Gradient Saliency", "Computes L1 norm of output logits with respect to WordPiece input embeddings:\n   S(w_i) = || ∂y_c / ∂e_i ||_1\nPinpoints provocative manipulation triggers (e.g. 'EXPLOSION', 'DESTROYED')."),
        ("2. Visual Layer4 Grad-CAM Overlays", "Global average pooling of gradients captures class-discriminative visual activation:\n   L_GradCAM = ReLU( sum_k α_k^c A^k )\nOverlays localized heatmaps on face manipulation seams and diffusion borders."),
        ("3. Acoustic Spectral Diagnostics", "High-frequency energy ratios (>4 kHz) and zero-crossing deviations isolate vocoder synthesis artifacts characteristic of ElevenLabs & Hume AI clones.")
    ]
    for x_title, x_desc in xai_cards:
        p1 = tf_r.add_paragraph()
        p1.text = x_title
        p1.font.bold = True
        p1.font.size = Pt(10)
        p1.font.color.rgb = COLOR_CYAN
        p1.space_before = Pt(8)

        p2 = tf_r.add_paragraph()
        p2.text = x_desc
        p2.font.size = Pt(8.8)
        p2.font.color.rgb = COLOR_WHITE
        p2.space_before = Pt(2)


def build_slide_7(prs):
    """SLIDE 7: Software Architecture & Production Tech Stack"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Software Architecture & Production Tech Stack", "06 · ENGINEERING & SYSTEM IMPLEMENTATION")

    # Left Column: 4-Tier Vertical Layered Architecture (65% width = 7.6 inches)
    w_stack = Inches(7.6)
    c_stack = create_card(slide, Inches(0.8), Inches(1.5), w_stack, Inches(5.4), border_color=COLOR_CYAN)
    tf_s = c_stack.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "FOUR-TIER PRODUCTION ARCHITECTURE"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_CYAN

    tiers = [
        ("TIER 1: CLIENT PRESENTATION (React 18 + TypeScript + Vite + Tailwind CSS)", [
            "Verification Workspace: Drag-and-drop text claims, images, MP4 videos, and audio waveforms",
            "Interactive Evaluation Hub: Real-time radar charts, ECE calibration curves, and multi-suite leaderboards",
            "History Audit Viewer: Chronological inspection of historical analyses with full XAI visualization"
        ]),
        ("TIER 2: ASYNCHRONOUS API GATEWAY (FastAPI + Uvicorn)", [
            "High-throughput async endpoints: POST /api/verify, GET /api/evaluation/metrics, GET /api/history",
            "Pydantic strict schema validation with robust multipart media payload ingestion"
        ]),
        ("TIER 3: MULTIMODAL ML INFERENCE RUNTIME (PyTorch 2.x + Hugging Face + Torchvision)", [
            "Modular inference pipeline with DistilBERT, ResNet18, 8-frame latent video pooling, and Librosa STFT",
            "Unified confidence calibrator with post-hoc temperature scaling (T* = 1.4628)"
        ]),
        ("TIER 4: PERSISTENCE & AUDIT COMPLIANCE (SQLite 3 - truthlens.db)", [
            "Zero-loss historical persistence: Stores multimodal inputs, prediction probabilities, and Grad-CAM arrays"
        ])
    ]

    for t_name, t_pts in tiers:
        p1 = tf_s.add_paragraph()
        p1.text = f"• {t_name}"
        p1.font.bold = True
        p1.font.size = Pt(9.5)
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(5)

        for pt in t_pts:
            p2 = tf_s.add_paragraph()
            p2.text = f"    - {pt}"
            p2.font.size = Pt(8.5)
            p2.font.color.rgb = COLOR_SLATE

    # Right Column: Tech Badges & Health Card (35% width = 3.9 inches)
    left_r = Inches(8.65)
    w_r = Inches(3.9)

    # Health Card (Top)
    c_health = create_card(slide, left_r, Inches(1.5), w_r, Inches(2.5), border_color=COLOR_GREEN, border_width=2)
    tf_h = c_health.text_frame
    tf_h.word_wrap = True
    tf_h.vertical_anchor = MSO_ANCHOR.MIDDLE
    ph1 = tf_h.paragraphs[0]
    ph1.text = "TEST SUITE HEALTH: 100%"
    ph1.font.bold = True
    ph1.font.size = Pt(13)
    ph1.font.color.rgb = COLOR_GREEN
    ph1.alignment = PP_ALIGN.CENTER

    ph2 = tf_h.add_paragraph()
    ph2.text = "96 / 96 Automated Tests Passing"
    ph2.font.bold = True
    ph2.font.size = Pt(16)
    ph2.font.color.rgb = COLOR_WHITE
    ph2.alignment = PP_ALIGN.CENTER
    ph2.space_before = Pt(4)

    ph3 = tf_h.add_paragraph()
    ph3.text = "• Full Unit & Integration Test Coverage\n• Zero TypeScript / JSX Build Errors\n• SQLite Zero-Loss Storage Verified\n• Audio/Video Gradient Backprop Confirmed"
    ph3.font.size = Pt(9)
    ph3.font.color.rgb = COLOR_SLATE
    ph3.alignment = PP_ALIGN.CENTER
    ph3.space_before = Pt(6)

    # Tech Stack Badges (Bottom)
    c_tech = create_card(slide, left_r, Inches(4.2), w_r, Inches(2.7), border_color=COLOR_BLUE)
    tf_t = c_tech.text_frame
    tf_t.word_wrap = True
    pt_t = tf_t.paragraphs[0]
    pt_t.text = "CORE TECHNOLOGIES"
    pt_t.font.bold = True
    pt_t.font.size = Pt(11)
    pt_t.font.color.rgb = COLOR_CYAN
    pt_t.alignment = PP_ALIGN.CENTER

    tech_pills = [
        "Python 3.13 · PyTorch 2.x",
        "Hugging Face Transformers",
        "Torchvision · Librosa · OpenCV",
        "FastAPI · Uvicorn · Pydantic",
        "React 18 · TypeScript · Vite",
        "Tailwind CSS · Lucide Icons",
        "SQLite 3 · Pytest Testing"
    ]
    for pill in tech_pills:
        p = tf_t.add_paragraph()
        p.text = f"• {pill}"
        p.font.size = Pt(9)
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(2)


def build_slide_8(prs):
    """SLIDE 8: Experimental Results & Baselines"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Experimental Results & Systematic Baseline Comparison", "07 · BENCHMARK PERFORMANCE")

    # Top Table: Baselines vs Fusion
    headers = ["Model Architecture", "Modalities", "In-Domain Acc.", "Fakeddit Acc.", "Audio ROC-AUC", "Video ROC-AUC", "Mean ECE"]
    rows = [
        ["DistilBERT Baseline", "Text Only", "100.0%", "33.33%", "—", "—", "0.2840"],
        ["ResNet18 Baseline", "Vision Only", "100.0%", "50.00%", "—", "—", "0.2210"],
        ["Acoustic Spectral Baseline", "Audio Only", "100.0%", "—", "0.8100", "—", "0.1840"],
        ["Video Temporal Pooling", "Video Only", "100.0%", "—", "—", "0.8200", "0.1790"],
        ["Concatenation Fusion", "Text + Vision", "100.0%", "43.33%", "—", "—", "0.1870"],
        ["TruthLens Cross-Attention", "T + V + A", "100.0%", "46.67%", "0.8200", "0.8200", "0.1324"]
    ]
    col_widths = [Inches(2.5), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.4), Inches(1.4), Inches(1.2)]
    add_styled_table(slide, Inches(0.8), Inches(1.5), Inches(11.75), Inches(2.9), headers, rows, col_widths, highlight_row_idx=5)

    # Bottom Callout Cards (3 Cards)
    card_w = Inches(3.75)
    gap = Inches(0.25)
    top_pos = Inches(4.7)
    h_card = Inches(2.2)

    callouts = [
        ("IN-DOMAIN SATURATION", "100.0%", "All evaluated baselines achieve perfect 100.0% accuracy on standard in-domain test splits. This confirms that clean laboratory evaluations fail to expose real-world vulnerabilities.", COLOR_AMBER),
        ("UNIMODAL TEXT COLLAPSE", "33.33%", "DistilBERT text branch collapses to 33.33% accuracy on uncurated Reddit slang (Fakeddit), proving that language models cannot verify content in isolation without cross-modal context.", COLOR_RED),
        ("CALIBRATION SUPERIORITY", "0.1324 ECE", "TruthLens bidirectional cross-attention achieves the lowest Expected Calibration Error across all models (0.1324 vs. 0.2840 for text), yielding dependable real-world probability estimates.", COLOR_GREEN)
    ]

    for idx, (title, stat, desc, col_acc) in enumerate(callouts):
        left_pos = Inches(0.8) + idx * (card_w + gap)
        c = create_card(slide, left_pos, top_pos, card_w, h_card, border_color=col_acc, border_width=1.5)
        tf = c.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = col_acc

        p2 = tf.add_paragraph()
        p2.text = stat
        p2.font.bold = True
        p2.font.size = Pt(20)
        p2.font.color.rgb = COLOR_WHITE
        p2.space_before = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(8.8)
        p3.font.color.rgb = COLOR_SLATE
        p3.space_before = Pt(4)


def build_slide_9(prs):
    """SLIDE 9: Semantic Mismatch, Calibration & Ablation Matrix"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Semantic Mismatch, Probability Calibration & Ablation Matrix", "08 · RIGOROUS ABLATION & CALIBRATION")

    # Left Column: Visual Dominance Trap & Calibration (45% width = 5.2 inches)
    w_left = Inches(5.2)
    c_left = create_card(slide, Inches(0.8), Inches(1.5), w_left, Inches(5.4), border_color=COLOR_AMBER)
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "THE VISUAL DOMINANCE TRAP & CALIBRATION"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_AMBER

    points_trap = [
        ("The Cheapfake Attack Vector:", "Authentic, unmanipulated photograph paired with a fabricated news headline (Condition 2)."),
        ("Baseline Failure (Config A):", "Achieved 0.0% accuracy on Condition 2! The high-capacity vision backbone dominated, convincing the model that authentic visual pixels imply an authentic overall claim."),
        ("Contrastive Alignment Fix (Config C):", "InfoNCE loss (λ = 0.05) repels mismatched pairs in latent space, restoring Condition 2 detection to 100.0% (+100.0% net gain!)."),
        ("Post-Hoc Probability Calibration:", "Uncalibrated models showed severe overconfidence on social media (ECE = 0.3406).\nTemperature scaling (T* = 1.4628) reduced ECE to 0.2701, routing 38.3% of borderline cases to UNCERTAIN.")
    ]
    for pt_t, pt_d in points_trap:
        p1 = tf_l.add_paragraph()
        p1.text = f"• {pt_t}"
        p1.font.bold = True
        p1.font.size = Pt(9.5)
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(4)

        p2 = tf_l.add_paragraph()
        p2.text = f"  {pt_d}"
        p2.font.size = Pt(8.6)
        p2.font.color.rgb = COLOR_SLATE

    # Right Column: Six-Configuration Ablation Matrix (55% width = 6.4 inches)
    headers = ["Configuration", "In-Dom.", "Mismatch", "Fakeddit", "A/V AUC", "ECE", "Composite Score"]
    rows = [
        ["Config A (Production)", "100.0%", "55.0%", "46.7%", "0.8200", "0.1324", "8.32 / 10 (WINNER)"],
        ["Config D (Calibrated)", "100.0%", "55.0%", "46.7%", "0.8200", "0.1338", "8.31 / 10"],
        ["Config C (Contrastive)", "100.0%", "60.0%", "46.7%", "0.8300", "0.1571", "8.26 / 10 (Cheapfake Def.)"],
        ["Config B (Adapted Text)", "100.0%", "73.3%", "56.7%", "0.8100", "0.2453", "8.01 / 10 (Calib. Drift)"],
        ["Config F (Full Proposed)", "100.0%", "48.3%", "46.7%", "0.8100", "0.1882", "7.82 / 10 (Neg. Transfer)"],
        ["Config E (Adapted+Contr)", "100.0%", "48.3%", "46.7%", "0.8100", "0.2104", "7.74 / 10 (Neg. Transfer)"]
    ]
    col_widths = [Inches(1.8), Inches(0.7), Inches(0.8), Inches(0.8), Inches(0.75), Inches(0.75), Inches(1.3)]
    add_styled_table(slide, Inches(6.2), Inches(1.5), Inches(6.35), Inches(3.6), headers, rows, col_widths, highlight_row_idx=0)

    # Scientific Caution Alert Box
    alert_box = create_card(slide, Inches(6.2), Inches(5.3), Inches(6.35), Inches(1.6), bg_color=RGBColor(35, 15, 25), border_color=COLOR_RED, border_width=1.5)
    tf_a = alert_box.text_frame
    tf_a.word_wrap = True
    tf_a.vertical_anchor = MSO_ANCHOR.MIDDLE
    pa1 = tf_a.paragraphs[0]
    pa1.text = "CRITICAL SCIENTIFIC FINDING: AVOID NEGATIVE TRANSFER"
    pa1.font.bold = True
    pa1.font.size = Pt(10)
    pa1.font.color.rgb = COLOR_RED
    pa2 = tf_a.add_paragraph()
    pa2.text = "Blindly stacking all proposed improvements (Config F: text adaptation + contrastive alignment) triggers negative transfer drift, dropping semantic mismatch detection to 48.3% and degrading calibration. Config A remains the scientifically verified production configuration."
    pa2.font.size = Pt(8.8)
    pa2.font.color.rgb = COLOR_WHITE


def build_slide_10(prs):
    """SLIDE 10: End-to-End Demo, Limitations & Future Roadmap"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "End-to-End Workflow, Honest Limitations & Future Roadmap", "09 · PRODUCTION DEMO & FUTURE WORK")

    col_w = Inches(3.75)
    gap = Inches(0.25)
    top_pos = Inches(1.5)
    h_col = Inches(4.5)

    # Column 1: Verification Workflow (5 Steps)
    c1 = create_card(slide, Inches(0.8), top_pos, col_w, h_col, border_color=COLOR_CYAN)
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "VERIFICATION WORKFLOW"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_CYAN

    steps = [
        ("1. Ingest Media:", "Text claim + Image, Video, or Audio file."),
        ("2. Deep Inference:", "DistilBERT, ResNet18, 8-frame pooling, STFT."),
        ("3. Calibrated Verdict:", "Applies τ = 0.65 threshold (REAL / FAKE / UNCERTAIN)."),
        ("4. Explainability Engine:", "Generates Grad-CAM heatmaps & token saliency."),
        ("5. SQLite Persistence:", "Full audit payload saved to truthlens.db.")
    ]
    for s_t, s_d in steps:
        p1 = tf1.add_paragraph()
        p1.text = f"• {s_t}"
        p1.font.bold = True
        p1.font.size = Pt(9.2)
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(3)

        p2 = tf1.add_paragraph()
        p2.text = f"  {s_d}"
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = COLOR_SLATE

    # Result Badge Preview
    r_badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(5.2), Inches(3.35), Inches(0.65))
    r_badge.fill.solid()
    r_badge.fill.fore_color.rgb = RGBColor(12, 23, 49)
    r_badge.line.color.rgb = COLOR_RED
    tf_rb = r_badge.text_frame
    tf_rb.vertical_anchor = MSO_ANCHOR.MIDDLE
    prb = tf_rb.paragraphs[0]
    prb.text = "VERDICT: FAKE  [Conf: 87.4% · Conflict: 0.82]"
    prb.font.bold = True
    prb.font.size = Pt(9.5)
    prb.font.color.rgb = COLOR_RED
    prb.alignment = PP_ALIGN.CENTER

    # Column 2: Honest Research Limitations
    c2 = create_card(slide, Inches(0.8) + col_w + gap, top_pos, col_w, h_col, border_color=COLOR_AMBER)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "HONEST RESEARCH LIMITATIONS"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_AMBER

    limits = [
        ("Dataset Scale Constraint:", "The primary balanced corpus is constrained to 320 samples; industrial deployment requires tens of thousands of diverse multimodal instances."),
        ("Absence of Video/Audio Labels:", "The primary training dataset contains zero labeled video and audio records, requiring the system to rely on zero-shot feature projections."),
        ("Calibration Trade-Offs:", "Rehearsal fine-tuning on noisy informal social media language induced noticeable calibration drift (ECE = 0.2453).")
    ]
    for l_t, l_d in limits:
        p1 = tf2.add_paragraph()
        p1.text = f"• {l_t}"
        p1.font.bold = True
        p1.font.size = Pt(9.2)
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(6)

        p2 = tf2.add_paragraph()
        p2.text = f"  {l_d}"
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = COLOR_SLATE

    # Column 3: Future Research Roadmap
    c3 = create_card(slide, Inches(0.8) + (col_w + gap)*2, top_pos, col_w, h_col, border_color=COLOR_BLUE)
    tf3 = c3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "FUTURE RESEARCH ROADMAP"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_BLUE

    roadmap = [
        ("Retrieval-Augmented Fact-Checking (RAG):", "Integrate external knowledge graphs, Wikipedia, and Google Fact Check APIs to cross-verify claims in real time."),
        ("Spatial Frequency Transforms (DCT/FFT):", "Incorporate Fourier and discrete cosine frequency analysis to expose subtle diffusion lattice artifacts."),
        ("Multilingual Cross-Attention:", "Extend language representations beyond English WordPiece to low-resource and regional Indic dialects.")
    ]
    for r_t, r_d in roadmap:
        p1 = tf3.add_paragraph()
        p1.text = f"• {r_t}"
        p1.font.bold = True
        p1.font.size = Pt(9.2)
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(6)

        p2 = tf3.add_paragraph()
        p2.text = f"  {r_d}"
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = COLOR_SLATE

    # Bottom Concluding Quote Banner
    banner = create_card(slide, Inches(0.8), Inches(6.15), Inches(11.75), Inches(0.85), bg_color=RGBColor(12, 23, 49), border_color=COLOR_CYAN, border_width=1.5)
    tf_ban = banner.text_frame
    tf_ban.word_wrap = True
    tf_ban.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_ban = tf_ban.paragraphs[0]
    p_ban.text = "“Deepfake verification is not an isolated computer vision problem; it is a cross-modal consistency and uncertainty reasoning challenge.”"
    p_ban.font.bold = True
    p_ban.font.size = Pt(11)
    p_ban.font.color.rgb = COLOR_CYAN
    p_ban.alignment = PP_ALIGN.CENTER


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    print("Building Slide 1: Title & Team Identity...")
    build_slide_1(prs)
    print("Building Slide 2: The Problem & Literature Gap...")
    build_slide_2(prs)
    print("Building Slide 3: Audited Datasets & Multi-Suite Setup...")
    build_slide_3(prs)
    print("Building Slide 4: Overall System Architecture...")
    build_slide_4(prs)
    print("Building Slide 5: Multimodal Methodology & Temporal Video Fix...")
    build_slide_5(prs)
    print("Building Slide 6: From Prediction to Evidence: Attention & XAI...")
    build_slide_6(prs)
    print("Building Slide 7: Software Architecture & Production Tech Stack...")
    build_slide_7(prs)
    print("Building Slide 8: Experimental Results & Baselines...")
    build_slide_8(prs)
    print("Building Slide 9: Semantic Mismatch, Calibration & Ablation Matrix...")
    build_slide_9(prs)
    print("Building Slide 10: End-to-End Demo, Limitations & Future Roadmap...")
    build_slide_10(prs)

    output_path = "TruthLens_AI_Presentation.pptx"
    prs.save(output_path)
    print(f"Successfully generated 10-slide presentation at {output_path}!")

if __name__ == "__main__":
    main()
