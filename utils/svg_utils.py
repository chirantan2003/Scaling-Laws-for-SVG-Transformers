"""
SVG parsing, cleaning, and normalization utilities.
"""
import re
from lxml import etree
from io import BytesIO
from typing import Optional, Tuple


# Namespaces commonly found in SVG files that should be stripped
STRIP_NAMESPACES = [
    "http://www.inkscape.org/namespaces/inkscape",
    "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd",
    "http://ns.adobe.com/AdobeIllustrator/10.0/",
    "http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://purl.org/dc/elements/1.1/",
    "http://creativecommons.org/ns#",
    "http://www.w3.org/2000/svg",  # We'll handle this specially
]

# Tags to remove entirely
STRIP_TAGS = [
    "metadata", "desc", "title",
    "{http://www.w3.org/2000/svg}metadata",
    "{http://www.w3.org/2000/svg}desc",
    "{http://www.w3.org/2000/svg}title",
]

# Regex to match floating point numbers (including negative, with optional decimal)
FLOAT_REGEX = re.compile(r'(?<![a-zA-Z])(-?\d+\.\d{2,})')


def parse_svg(svg_string: str) -> Optional[etree._Element]:
    """Parse an SVG string into an lxml Element tree.
    
    Returns None if parsing fails.
    """
    try:
        # Try parsing as bytes first (handles encoding declarations)
        if isinstance(svg_string, str):
            svg_bytes = svg_string.encode("utf-8")
        else:
            svg_bytes = svg_string
        
        parser = etree.XMLParser(
            remove_comments=True,
            remove_pis=True,
            recover=True,  # Try to recover from malformed XML
        )
        tree = etree.parse(BytesIO(svg_bytes), parser)
        return tree.getroot()
    except Exception:
        return None


def strip_metadata_elements(root: etree._Element) -> etree._Element:
    """Remove metadata, desc, title elements and editor-specific namespaced elements."""
    # Remove known metadata tags
    for tag in STRIP_TAGS:
        for elem in root.iter(tag):
            elem.getparent().remove(elem)
    
    # Remove elements from editor namespaces (Inkscape, Sodipodi, etc.)
    to_remove = []
    for elem in root.iter():
        tag = elem.tag if isinstance(elem.tag, str) else ""
        for ns in STRIP_NAMESPACES[:-1]:  # Skip SVG namespace itself
            if ns in tag:
                to_remove.append(elem)
                break
    
    for elem in to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)
    
    return root


def strip_namespace_prefixes(root: etree._Element) -> etree._Element:
    """Remove namespace prefixes from attributes (e.g., inkscape:label -> label)."""
    for elem in root.iter():
        # Clean namespace from tag
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
        
        # Clean namespace from attributes
        attribs_to_remove = []
        attribs_to_add = {}
        for key, value in elem.attrib.items():
            if "}" in key:
                clean_key = key.split("}", 1)[1]
                # Only keep if it's a meaningful SVG attribute
                # Skip editor-specific attributes
                if not any(ns in key for ns in STRIP_NAMESPACES[:-1]):
                    attribs_to_add[clean_key] = value
                attribs_to_remove.append(key)
            elif ":" in key:
                # Remove prefixed attributes like inkscape:version, sodipodi:docname
                prefix = key.split(":")[0]
                if prefix in ("inkscape", "sodipodi", "rdf", "dc", "cc", "ns"):
                    attribs_to_remove.append(key)
        
        for key in attribs_to_remove:
            del elem.attrib[key]
        for key, value in attribs_to_add.items():
            if key not in elem.attrib:
                elem.attrib[key] = value
    
    return root


def round_coordinates(svg_string: str, precision: int = 1) -> str:
    """Round floating-point numbers in SVG to the specified decimal precision.
    
    This reduces vocabulary size by normalizing near-duplicate float tokens.
    e.g., precision=1: 12.3456 -> 12.3
    """
    def _round_match(match):
        value = float(match.group(0))
        rounded = round(value, precision)
        # Format to remove trailing zeros but keep at least one decimal if needed
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.{precision}f}"
    
    return FLOAT_REGEX.sub(_round_match, svg_string)


def canonicalize_attributes(root: etree._Element) -> etree._Element:
    """Sort attributes alphabetically within each element for consistency."""
    for elem in root.iter():
        if len(elem.attrib) > 1:
            attribs = sorted(elem.attrib.items())
            # Clear and re-add in sorted order
            for key in list(elem.attrib.keys()):
                del elem.attrib[key]
            for key, value in attribs:
                elem.attrib[key] = value
    return root


def normalize_whitespace(svg_string: str) -> str:
    """Collapse multiple whitespace/newlines to single spaces, strip edges."""
    # Collapse whitespace
    svg_string = re.sub(r'\s+', ' ', svg_string)
    # Remove spaces around XML structural characters for compactness
    svg_string = re.sub(r'\s*>\s*<\s*', '><', svg_string)
    svg_string = re.sub(r'\s*/>', '/>', svg_string)
    return svg_string.strip()


def serialize_svg(root: etree._Element) -> str:
    """Serialize an lxml Element tree back to an SVG string."""
    # Remove any namespace declarations from root
    etree.cleanup_namespaces(root)
    
    svg_bytes = etree.tostring(
        root,
        pretty_print=False,
        xml_declaration=False,
        encoding="unicode",
    )
    return svg_bytes


def validate_xml(svg_string: str) -> bool:
    """Check if a string is valid XML."""
    try:
        etree.fromstring(svg_string.encode("utf-8"))
        return True
    except Exception:
        return False


def validate_render(svg_string: str) -> bool:
    """Attempt to render SVG to PNG using CairoSVG. Returns True if successful."""
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_string.encode("utf-8"))
        return True
    except Exception:
        return False


def render_svg_to_png(svg_string: str, output_path: str, width: int = 256) -> bool:
    """Render an SVG string to a PNG file.
    
    Uses svglib+reportlab (pure Python) as primary renderer.
    Falls back to CairoSVG if available.
    """
    # Try svglib + reportlab (pure Python, no native deps)
    try:
        import tempfile, os
        from svglib.svglib import renderSVG
        from reportlab.graphics import renderPM
        
        # svglib needs a file path, so write to temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".svg")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(svg_string)
            drawing = renderSVG.render(tmp_path)
            if drawing is None:
                raise ValueError("svglib returned None")
            # Scale to target width
            scale = width / max(drawing.width, drawing.height, 1)
            drawing.width = width
            drawing.height = width
            drawing.scale(scale, scale)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            return True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception:
        pass
    
    # Fallback: CairoSVG
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_string.encode("utf-8"),
            write_to=output_path,
            output_width=width,
            output_height=width,
        )
        return True
    except Exception as e:
        print(f"  [render] Failed to render: {e}")
        return False


def clean_svg(
    svg_string: str,
    coord_precision: int = 1,
    canonicalize: bool = True,
    min_length: int = 50,
    max_length: int = 50_000,
) -> Tuple[Optional[str], dict]:
    """
    Full SVG cleaning pipeline.
    
    Returns:
        (cleaned_svg_string or None, stats_dict)
    """
    stats = {
        "original_length": len(svg_string),
        "cleaned_length": 0,
        "valid_xml": False,
        "render_ok": None,
        "reject_reason": None,
    }
    
    # Length pre-check
    if len(svg_string) < min_length:
        stats["reject_reason"] = "too_short"
        return None, stats
    
    if len(svg_string) > max_length:
        stats["reject_reason"] = "too_long"
        return None, stats
    
    # Parse XML
    root = parse_svg(svg_string)
    if root is None:
        stats["reject_reason"] = "parse_failed"
        return None, stats
    
    # Strip metadata and editor elements
    root = strip_metadata_elements(root)
    
    # Strip namespace prefixes
    root = strip_namespace_prefixes(root)
    
    # Canonicalize attribute ordering
    if canonicalize:
        root = canonicalize_attributes(root)
    
    # Serialize back to string
    svg_cleaned = serialize_svg(root)
    
    # Round coordinates
    svg_cleaned = round_coordinates(svg_cleaned, precision=coord_precision)
    
    # Normalize whitespace
    svg_cleaned = normalize_whitespace(svg_cleaned)
    
    # Post-cleaning length check
    if len(svg_cleaned) < min_length:
        stats["reject_reason"] = "too_short_after_clean"
        return None, stats
    
    # Validate re-parsed XML
    if not validate_xml(svg_cleaned):
        stats["reject_reason"] = "invalid_xml_after_clean"
        return None, stats
    
    stats["valid_xml"] = True
    stats["cleaned_length"] = len(svg_cleaned)
    
    return svg_cleaned, stats
