"""
Standards Librarian MCP Server

A lightweight MCP server that helps Claude find and access regulatory standards PDFs.
Instead of parsing/chunking/embedding, this server:

1. Maintains a simple index of what standards you have
2. Helps Claude find the right standard(s) for a question
3. Returns PDF file paths for Claude to read directly

This leverages Claude's native PDF understanding - no complex RAG pipeline needed.

Usage:
    python -m src.mcp_librarian
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    CallToolRequest,
    ListToolsResult,
    Resource,
    ResourceContents,
    TextResourceContents,
    BlobResourceContents,
    ListResourcesResult,
    ReadResourceRequest,
    ReadResourceResult,
)
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("standards-librarian")

server = Server("standards-librarian")


# =============================================================================
# Standards Index (simple JSON-based)
# =============================================================================

@dataclass
class StandardInfo:
    """Metadata about a standard."""
    id: str                              # e.g., "IEC_60601-1"
    title: str                           # Full title
    short_title: str                     # e.g., "IEC 60601-1"
    filename: str                        # PDF filename
    description: str                     # What this standard covers
    scope: str                           # Brief scope description
    sections: dict[str, str]             # Section number -> description mapping
    related_standards: list              # List of related standard objects
    organization: str = "IEC"
    year: str = ""
    pages: int = 0
    annexes: dict[str, dict] = field(default_factory=dict)       # Annex ID -> {description, normative, related_sections}
    key_terms: list[str] = field(default_factory=list)           # Defined terms from Section 3
    key_tables: dict[str, dict] = field(default_factory=dict)    # Table ID -> {description, location, related_sections}
    key_figures: dict[str, dict] = field(default_factory=dict)   # Figure ID -> {description, location, related_sections}
    notes: str = ""                                               # Extraction notes/limitations
    key_topics: list[str] = field(default_factory=list)          # Fallback search terms (optional)
    
    def matches_query(self, query: str) -> tuple[bool, float]:
        """Check if this standard is relevant to a query. Returns (match, score)."""
        query_lower = query.lower()
        score = 0.0
        
        # Check title
        if query_lower in self.title.lower():
            score += 3.0
        
        # Check description
        if query_lower in self.description.lower():
            score += 2.0
        
        # Check scope
        if query_lower in self.scope.lower():
            score += 2.0
        
        # Check key topics (fallback)
        for topic in self.key_topics:
            if query_lower in topic.lower() or topic.lower() in query_lower:
                score += 1.5
        
        # Check key terms (defined terms from Section 3)
        for term in self.key_terms:
            if query_lower in term.lower() or term.lower() in query_lower:
                score += 1.5
        
        # Check section descriptions
        for section, desc in self.sections.items():
            if query_lower in desc.lower():
                score += 1.0
        
        # Check annex descriptions
        for annex, annex_data in self.annexes.items():
            if query_lower in annex_data['description'].lower():
                score += 0.8
        
        # Check table descriptions
        for table, table_data in self.key_tables.items():
            if query_lower in table_data['description'].lower():
                score += 1.0
        
        # Check figure descriptions
        for figure, figure_data in self.key_figures.items():
            if query_lower in figure_data['description'].lower():
                score += 0.5
        
        # Check individual words
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3:  # Skip short words
                if word in self.title.lower():
                    score += 0.5
                if word in self.description.lower():
                    score += 0.3
                if any(word in topic.lower() for topic in self.key_topics):
                    score += 0.5
                if any(word in term.lower() for term in self.key_terms):
                    score += 0.5
        
        return (score > 0, score)


@dataclass
class CrossReference:
    """A cross-reference entry mapping a topic to standards/sections."""
    topic: str                           # The topic name (e.g., "leakage current")
    aliases: list[str]                   # Alternative names/spellings
    primary_standard: str                # Primary standard ID
    primary_section: str                 # Primary section within that standard
    primary_note: str = ""               # Brief note about what's there
    also_see: list[dict] = field(default_factory=list)  # [{standard, section, note}]


@dataclass
class StandardsLibrary:
    """The library of available standards and cross-references."""
    standards: dict[str, StandardInfo] = field(default_factory=dict)
    cross_references: dict[str, CrossReference] = field(default_factory=dict)  # topic -> CrossReference
    pdf_directory: str = "./data/pdfs"
    
    def add_standard(self, standard: StandardInfo):
        """Add a standard to the library."""
        self.standards[standard.id] = standard
    
    def add_cross_reference(self, xref: CrossReference):
        """Add a cross-reference entry."""
        # Index by topic and all aliases
        self.cross_references[xref.topic.lower()] = xref
        for alias in xref.aliases:
            self.cross_references[alias.lower()] = xref
    
    def lookup_topic(self, query: str) -> Optional[CrossReference]:
        """Look up a topic in cross-references. Returns None if not found."""
        query_lower = query.lower()
        
        # Exact match
        if query_lower in self.cross_references:
            return self.cross_references[query_lower]
        
        # Partial match - check if query is contained in any topic
        for topic, xref in self.cross_references.items():
            if query_lower in topic or topic in query_lower:
                return xref
        
        return None
    
    def find_standards(self, query: str, limit: int = 3) -> list[tuple[StandardInfo, float]]:
        """Find standards relevant to a query (fallback search)."""
        results = []
        for std in self.standards.values():
            matches, score = std.matches_query(query)
            if matches:
                results.append((std, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_pdf_path(self, standard_id: str) -> Optional[Path]:
        """Get the full path to a standard's PDF."""
        std = self.standards.get(standard_id)
        if std:
            path = Path(self.pdf_directory) / std.filename
            if path.exists():
                return path
        return None
    
    def save(self, path: str = "data/standards_index.json"):
        """Save the library index to JSON."""
        # Convert cross_references to serializable format (dedupe aliases)
        xrefs_data = {}
        seen_topics = set()
        for topic, xref in self.cross_references.items():
            if xref.topic not in seen_topics:
                seen_topics.add(xref.topic)
                xrefs_data[xref.topic] = {
                    "aliases": xref.aliases,
                    "primary_standard": xref.primary_standard,
                    "primary_section": xref.primary_section,
                    "primary_note": xref.primary_note,
                    "also_see": xref.also_see,
                }
        
        data = {
            "pdf_directory": self.pdf_directory,
            "standards": {k: asdict(v) for k, v in self.standards.items()},
            "cross_references": xrefs_data,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str = "data/standards_index.json") -> "StandardsLibrary":
        """Load the library index from JSON."""
        library = cls()
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            library.pdf_directory = data.get("pdf_directory", "./data/pdfs")
            
            # Load standards
            for std_id, std_data in data.get("standards", {}).items():
                library.standards[std_id] = StandardInfo(**std_data)
            
            # Load cross-references
            for topic, xref_data in data.get("cross_references", {}).items():
                xref = CrossReference(
                    topic=topic,
                    aliases=xref_data.get("aliases", []),
                    primary_standard=xref_data["primary_standard"],
                    primary_section=xref_data["primary_section"],
                    primary_note=xref_data.get("primary_note", ""),
                    also_see=xref_data.get("also_see", []),
                )
                library.add_cross_reference(xref)
        
        return library


# =============================================================================
# Global Library Instance
# =============================================================================

library: Optional[StandardsLibrary] = None


def get_index_path() -> str:
    """Get index path from environment or default."""
    return os.environ.get("STANDARDS_INDEX_PATH", "data/standards_index.json")


def get_pdf_directory() -> str:
    """Get PDF directory from environment or default."""
    return os.environ.get("STANDARDS_PDF_DIR", "data/pdfs")


def initialize():
    """Initialize the library."""
    global library
    
    index_path = get_index_path()
    pdf_dir = get_pdf_directory()
    
    logger.info(f"Loading standards index from: {index_path}")
    library = StandardsLibrary.load(index_path)
    library.pdf_directory = pdf_dir
    
    # If empty, create example entries
    if not library.standards:
        logger.info("No standards index found. Creating example entries.")
        create_example_index()
    
    logger.info(f"Loaded {len(library.standards)} standards")


def create_example_index():
    """Create example index entries for common medical device standards."""
    global library
    
    # IEC 60601-1
    library.add_standard(StandardInfo(
        id="IEC_60601-1",
        title="Medical electrical equipment – Part 1: General requirements for basic safety and essential performance",
        short_title="IEC 60601-1",
        filename="IEC_60601-1.pdf",
        description="General safety standard for medical electrical equipment. Covers electrical safety, mechanical safety, radiation safety, and risk management requirements for all medical devices that are electrically powered.",
        scope="Applies to basic safety and essential performance of medical electrical equipment (ME equipment) and medical electrical systems (ME systems).",
        key_topics=[
            "electrical safety",
            "patient leakage current",
            "applied parts",
            "Type B", "Type BF", "Type CF",
            "means of protection",
            "creepage distance",
            "air clearance",
            "protective earth",
            "single fault condition",
            "normal condition",
            "risk management",
            "essential performance",
            "basic safety",
            "enclosure",
            "temperature limits",
            "mechanical hazards",
            "biocompatibility",
            "cleaning and sterilization",
            "electromagnetic compatibility",
            "programmable electrical medical systems",
            "PEMS",
            "software",
            "usability",
            "alarms",
            "marking and labeling",
        ],
        sections={
            "1": "Scope, object and related standards",
            "3": "Terms and definitions",
            "4": "General requirements (risk management, essential performance)",
            "5": "General requirements for testing",
            "6": "Classification of ME equipment and ME systems",
            "7": "Identification, marking and documents",
            "8": "Protection against electrical hazards - leakage currents, dielectric strength, creepage/clearance",
            "9": "Protection against mechanical hazards",
            "10": "Protection against unwanted and excessive radiation hazards",
            "11": "Protection against excessive temperatures and other hazards",
            "12": "Accuracy of controls and instruments and protection against hazardous outputs",
            "13": "Hazardous situations and fault conditions",
            "14": "Programmable electrical medical systems (PEMS)",
            "15": "Construction of ME equipment",
            "16": "ME systems",
            "17": "Electromagnetic compatibility",
        },
        annexes={
            "Annex A": {
                "description": "General requirements, tests and guidance for alarm systems in ME equipment",
                "normative": True,
                "related_sections": ["12"]
            },
            "Annex B": {
                "description": "General requirements, tests and guidance for ME systems",
                "normative": True,
                "related_sections": ["16"]
            },
            "Annex F": {
                "description": "Test methods for leakage currents and patient auxiliary currents",
                "normative": True,
                "related_sections": ["8.7"]
            },
            "Annex H": {
                "description": "Rationale for PEMS requirements - software safety guidance",
                "normative": False,
                "related_sections": ["14"]
            },
            "Annex J": {
                "description": "Rationale for electrical safety requirements",
                "normative": False,
                "related_sections": ["8"]
            },
        },
        key_terms=[
            "APPLIED PART",
            "BASIC SAFETY",
            "ESSENTIAL PERFORMANCE",
            "LEAKAGE CURRENT",
            "PATIENT LEAKAGE CURRENT",
            "TOUCH CURRENT",
            "EARTH LEAKAGE CURRENT",
            "MEANS OF OPERATOR PROTECTION",
            "MEANS OF PATIENT PROTECTION",
            "SINGLE FAULT CONDITION",
            "NORMAL CONDITION",
            "TYPE B APPLIED PART",
            "TYPE BF APPLIED PART",
            "TYPE CF APPLIED PART",
            "PEMS",
            "ME EQUIPMENT",
            "ME SYSTEM",
        ],
        key_tables={
            "Table 1": {
                "description": "Classification of APPLIED PARTS - Type B, BF, CF symbols and descriptions",
                "location": "6.3",
                "related_sections": ["8.7", "8.5"]
            },
            "Table 3": {
                "description": "Allowable values of PATIENT LEAKAGE CURRENT and PATIENT AUXILIARY CURRENT - NC and SFC limits",
                "location": "8.7.3",
                "related_sections": ["8.7.4", "Annex F"]
            },
            "Table 4": {
                "description": "Allowable values of TOUCH CURRENT and EARTH LEAKAGE CURRENT",
                "location": "8.7.3",
                "related_sections": ["Annex F"]
            },
            "Table 6": {
                "description": "Creepage distances and air clearances - MOOP values",
                "location": "8.9",
                "related_sections": ["8.8"]
            },
            "Table 10": {
                "description": "Maximum temperatures of applied parts and surfaces",
                "location": "11.1",
                "related_sections": []
            },
        },
        key_figures={
            "Figure 1": {
                "description": "Relationship of standards in the IEC 60601 series",
                "location": "1",
                "related_sections": []
            },
            "Figure 3": {
                "description": "Classification decision tree for applied parts",
                "location": "6.3",
                "related_sections": ["8.7"]
            },
            "Figure F.1": {
                "description": "Test circuit for measurement of PATIENT LEAKAGE CURRENT - Type B applied part",
                "location": "Annex F",
                "related_sections": ["8.7.3", "8.7.4"]
            },
            "Figure F.2": {
                "description": "Test circuit for measurement of PATIENT LEAKAGE CURRENT - Type BF applied part",
                "location": "Annex F",
                "related_sections": ["8.7.3", "8.7.4"]
            },
            "Figure H.1": {
                "description": "Overview of PEMS development process",
                "location": "Annex H",
                "related_sections": ["14"]
            },
        },
        related_standards=[
            {
                "id": "ISO_14971",
                "relationship": "normative_reference",
                "description": "Risk management - required for clause 4 compliance"
            },
            {
                "id": "IEC_62304",
                "relationship": "gap_coverage",
                "description": "Software lifecycle - referenced by clause 14 (PEMS) for detailed software requirements"
            },
            {
                "id": "IEC_60601-1-2",
                "relationship": "collateral_standard",
                "description": "EMC requirements - detailed electromagnetic compatibility requirements for clause 17"
            },
            {
                "id": "IEC_60601-1-6",
                "relationship": "collateral_standard",
                "description": "Usability - detailed usability engineering requirements"
            },
            {
                "id": "IEC_60601-1-8",
                "relationship": "collateral_standard",
                "description": "Alarm systems - detailed requirements supplementing Annex A"
            },
        ],
        organization="IEC",
        year="2005+AMD1:2012",
        pages=500,
    ))
    
    # ISO 14708-1
    library.add_standard(StandardInfo(
        id="ISO_14708-1",
        title="Implants for surgery — Active implantable medical devices — Part 1: General requirements for safety, marking and for information to be provided by the manufacturer",
        short_title="ISO 14708-1",
        filename="ISO_14708-1.pdf",
        description="Specific requirements for active implantable medical devices (AIMDs) such as pacemakers, defibrillators, neurostimulators, and implantable drug pumps. Supplements IEC 60601-1 with implant-specific requirements.",
        scope="Applies to active implantable medical devices intended to be totally or partially introduced into the human body.",
        key_topics=[
            "active implantable medical device",
            "AIMD",
            "pacemaker",
            "defibrillator",
            "ICD",
            "neurostimulator",
            "implantable pump",
            "cochlear implant",
            "implant safety",
            "biocompatibility",
            "sterility",
            "packaging",
            "shelf life",
            "implant longevity",
            "battery life",
            "hermeticity",
            "MRI safety",
            "electromagnetic immunity",
            "wireless telemetry",
            "patient programmer",
            "clinician programmer",
        ],
        sections={
            "1": "Scope",
            "3": "Terms and definitions",
            "4": "General requirements",
            "5": "Protection against electrical hazards",
            "6": "Protection against mechanical hazards",
            "7": "Protection against radiation hazards",
            "8": "Protection against excessive temperatures",
            "9": "Protection against hazards from energy and substance delivery",
            "10": "Environmental conditions",
            "11": "Biocompatibility",
            "12": "Sterility",
            "13": "Instructions for use and labeling",
        },
        annexes={
            "Annex A": {
                "description": "Rationale for requirements",
                "normative": False,
                "related_sections": ["general"]
            },
            "Annex B": {
                "description": "Test methods for hermeticity",
                "normative": True,
                "related_sections": ["6"]
            },
        },
        key_terms=[
            "ACTIVE IMPLANTABLE MEDICAL DEVICE",
            "AIMD",
            "IMPLANTABLE PART",
            "NON-IMPLANTABLE PART",
            "PROGRAMMER",
            "THERAPEUTIC OUTPUT",
        ],
        key_tables={
            "Table 1": {
                "description": "Environmental conditions for storage and transport",
                "location": "10",
                "related_sections": []
            },
        },
        key_figures={
            "Figure 1": {
                "description": "Example AIMD system showing implantable and non-implantable parts",
                "location": "3",
                "related_sections": ["4", "5"]
            },
        },
        related_standards=[
            {
                "id": "IEC_60601-1",
                "relationship": "parent_standard",
                "description": "General safety requirements - ISO 14708-1 modifies and supplements 60601-1 for implants"
            },
            {
                "id": "ISO_14971",
                "relationship": "normative_reference",
                "description": "Risk management process"
            },
            {
                "id": "ISO_10993-1",
                "relationship": "normative_reference",
                "description": "Biocompatibility evaluation - required for clause 11"
            },
            {
                "id": "IEC_62304",
                "relationship": "normative_reference",
                "description": "Software lifecycle for AIMD software"
            },
        ],
        organization="ISO",
        year="2014",
        pages=100,
    ))
    
    # ISO 14971
    library.add_standard(StandardInfo(
        id="ISO_14971",
        title="Medical devices — Application of risk management to medical devices",
        short_title="ISO 14971",
        filename="ISO_14971.pdf",
        description="The fundamental risk management standard for medical devices. Defines the process for identifying hazards, estimating and evaluating risks, controlling risks, and monitoring effectiveness.",
        scope="Applies to all stages of the medical device lifecycle. Applicable to any medical device.",
        key_topics=[
            "risk management",
            "risk analysis",
            "risk evaluation",
            "risk control",
            "hazard identification",
            "harm",
            "severity",
            "probability",
            "risk estimation",
            "risk acceptability",
            "ALARP",
            "benefit-risk",
            "residual risk",
            "risk management file",
            "risk management plan",
            "risk management report",
            "foreseeable misuse",
            "intended use",
            "reasonably foreseeable misuse",
            "FMEA",
            "fault tree",
            "hazard analysis",
        ],
        sections={
            "1": "Scope",
            "3": "Terms and definitions - 26 defined terms including harm, hazard, risk, severity",
            "4": "General requirements for risk management - process, plan, file, competence",
            "5": "Risk analysis - intended use, hazard identification, risk estimation",
            "6": "Risk evaluation - criteria for risk acceptability",
            "7": "Risk control - option analysis, implementation, residual risk, benefit-risk",
            "8": "Evaluation of overall residual risk",
            "9": "Risk management review",
            "10": "Production and post-production activities",
        },
        annexes={
            "Annex A": {
                "description": "Rationale for requirements - explains reasoning behind each clause",
                "normative": False,
                "related_sections": ["4", "5", "6", "7", "8", "9", "10"]
            },
            "Annex B": {
                "description": "Risk management process overview - flowcharts and process description",
                "normative": False,
                "related_sections": ["4", "general"]
            },
            "Annex C": {
                "description": "Questions for identifying characteristics that could impact safety - hazard identification prompts",
                "normative": False,
                "related_sections": ["5"]
            },
        },
        key_terms=[
            "HARM",
            "HAZARD",
            "HAZARDOUS SITUATION",
            "RISK",
            "SEVERITY",
            "PROBABILITY OF OCCURRENCE",
            "RISK ANALYSIS",
            "RISK ASSESSMENT",
            "RISK CONTROL",
            "RISK ESTIMATION",
            "RISK EVALUATION",
            "RISK MANAGEMENT",
            "RISK MANAGEMENT FILE",
            "RESIDUAL RISK",
            "BENEFIT-RISK ANALYSIS",
            "INTENDED USE",
            "REASONABLY FORESEEABLE MISUSE",
        ],
        key_tables={},
        key_figures={
            "Figure 1": {
                "description": "Schematic representation of the risk management process",
                "location": "4",
                "related_sections": ["5", "6", "7"]
            },
            "Figure B.1": {
                "description": "Risk management process flowchart - complete overview",
                "location": "Annex B",
                "related_sections": ["4"]
            },
            "Figure B.2": {
                "description": "Risk analysis process flowchart",
                "location": "Annex B",
                "related_sections": ["5"]
            },
            "Figure B.3": {
                "description": "Risk control process flowchart",
                "location": "Annex B",
                "related_sections": ["7"]
            },
        },
        related_standards=[
            {
                "id": "IEC_60601-1",
                "relationship": "overlapping",
                "description": "Medical electrical equipment - requires ISO 14971 compliance, applies risk to electrical hazards"
            },
            {
                "id": "IEC_62304",
                "relationship": "overlapping",
                "description": "Medical device software - requires ISO 14971 for software risk management and safety classification"
            },
            {
                "id": "ISO_13485",
                "relationship": "normative_reference",
                "description": "Quality management systems - requires risk-based approach, references ISO 14971"
            },
            {
                "id": "ISO_TR_24971",
                "relationship": "informative_reference",
                "description": "Guidance on application - technical report with detailed guidance on applying ISO 14971"
            },
        ],
        organization="ISO",
        year="2019",
        pages=40,
    ))
    
    # IEC 62304
    library.add_standard(StandardInfo(
        id="IEC_62304",
        title="Medical device software – Software life cycle processes",
        short_title="IEC 62304",
        filename="IEC_62304.pdf",
        description="Software lifecycle standard for medical device software. Defines development, maintenance, risk management, configuration management, and problem resolution processes based on software safety classification.",
        scope="Applies to development and maintenance of medical device software. Covers software as a medical device (SaMD) and software in a medical device.",
        key_topics=[
            "software lifecycle",
            "software development",
            "software safety classification",
            "Class A", "Class B", "Class C",
            "software requirements",
            "software architecture",
            "software design",
            "software unit",
            "software integration",
            "software testing",
            "software verification",
            "software validation",
            "software configuration management",
            "software problem resolution",
            "software maintenance",
            "SOUP",
            "software of unknown provenance",
            "off-the-shelf software",
            "OTS",
            "traceability",
            "software anomaly",
            "regression testing",
        ],
        sections={
            "1": "Scope",
            "3": "Terms and definitions",
            "4": "General requirements - quality management, risk management, software safety classification",
            "5": "Software development process - planning, requirements, architecture, design, unit implementation, integration, testing",
            "6": "Software maintenance process",
            "7": "Software risk management process - hazard analysis, risk control, verification",
            "8": "Software configuration management process",
            "9": "Software problem resolution process",
        },
        annexes={
            "Annex A": {
                "description": "Rationale for requirements",
                "normative": False,
                "related_sections": ["general"]
            },
            "Annex B": {
                "description": "Guidance on provisions of this standard - detailed implementation guidance",
                "normative": False,
                "related_sections": ["4", "5", "6", "7", "8", "9"]
            },
            "Annex C": {
                "description": "Relationship to other standards - mapping to IEC 60601-1, ISO 14971",
                "normative": False,
                "related_sections": ["general"]
            },
        },
        key_terms=[
            "SOFTWARE SAFETY CLASS",
            "CLASS A",
            "CLASS B", 
            "CLASS C",
            "SOUP",
            "SOFTWARE UNIT",
            "SOFTWARE ITEM",
            "SOFTWARE SYSTEM",
            "SOFTWARE ARCHITECTURE",
            "TRACEABILITY",
            "SOFTWARE ANOMALY",
            "SOFTWARE PROBLEM REPORT",
        ],
        key_tables={
            "Table A.1": {
                "description": "Software safety classification - determines required activities based on risk",
                "location": "4.3",
                "related_sections": ["5", "7"]
            },
            "Table A.2": {
                "description": "Activities required by software safety class",
                "location": "Annex A",
                "related_sections": ["4.3", "5"]
            },
        },
        key_figures={
            "Figure 1": {
                "description": "Software development process overview",
                "location": "5",
                "related_sections": ["4"]
            },
            "Figure 2": {
                "description": "Relationship between software items, units, and systems",
                "location": "3",
                "related_sections": ["5"]
            },
        },
        related_standards=[
            {
                "id": "ISO_14971",
                "relationship": "normative_reference",
                "description": "Risk management - required for software safety classification and risk control"
            },
            {
                "id": "IEC_60601-1",
                "relationship": "overlapping",
                "description": "Medical electrical equipment - clause 14 (PEMS) references IEC 62304 for software"
            },
            {
                "id": "IEC_82304-1",
                "relationship": "overlapping",
                "description": "Health software - general requirements for standalone health software"
            },
            {
                "id": "ISO_13485",
                "relationship": "overlapping",
                "description": "Quality management - design control requirements apply to software development"
            },
        ],
        organization="IEC",
        year="2006+AMD1:2015",
        pages=80,
    ))
    
    # ==========================================================================
    # Cross-References - Quick lookup for common topics
    # ==========================================================================
    
    library.add_cross_reference(CrossReference(
        topic="leakage current",
        aliases=["patient leakage current", "leakage current limits", "touch current", "earth leakage"],
        primary_standard="IEC_60601-1",
        primary_section="8.7",
        primary_note="Allowable values in Table 3 and Table 4. Test methods in Annex F.",
        also_see=[
            {"standard": "IEC_60601-1", "section": "Annex F", "note": "Test circuits and measurement methods"},
            {"standard": "ISO_14708-1", "section": "5", "note": "Implant-specific electrical requirements"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="software safety classification",
        aliases=["software class", "Class A", "Class B", "Class C", "safety classification"],
        primary_standard="IEC_62304",
        primary_section="4.3",
        primary_note="Classification based on severity of harm. Determines required activities.",
        also_see=[
            {"standard": "IEC_60601-1", "section": "14", "note": "PEMS requirements reference 62304"},
            {"standard": "ISO_14971", "section": "5", "note": "Risk analysis informs classification"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="risk management",
        aliases=["risk analysis", "hazard analysis", "risk control", "risk assessment"],
        primary_standard="ISO_14971",
        primary_section="4-10",
        primary_note="Complete risk management process. Sections 4-10 cover plan through post-production.",
        also_see=[
            {"standard": "IEC_60601-1", "section": "4", "note": "Risk management requirements for ME equipment"},
            {"standard": "IEC_62304", "section": "7", "note": "Software-specific risk management"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="essential performance",
        aliases=["EP", "clinical function"],
        primary_standard="IEC_60601-1",
        primary_section="4.3",
        primary_note="Performance necessary to avoid unacceptable risk. Manufacturer-defined.",
        also_see=[
            {"standard": "ISO_14971", "section": "5", "note": "Risk analysis identifies essential performance"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="applied part",
        aliases=["applied parts", "Type B", "Type BF", "Type CF", "patient connection"],
        primary_standard="IEC_60601-1",
        primary_section="6.3",
        primary_note="Classification in Table 1. Affects leakage current limits.",
        also_see=[
            {"standard": "IEC_60601-1", "section": "8.7", "note": "Leakage limits by applied part type"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="biocompatibility",
        aliases=["biocompatible", "biological evaluation", "ISO 10993"],
        primary_standard="IEC_60601-1",
        primary_section="11.7",
        primary_note="References ISO 10993-1 for biological evaluation.",
        also_see=[
            {"standard": "ISO_14708-1", "section": "11", "note": "Implant-specific biocompatibility"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="SOUP",
        aliases=["software of unknown provenance", "OTS", "off-the-shelf software", "third-party software"],
        primary_standard="IEC_62304",
        primary_section="5.3",
        primary_note="Requirements for using SOUP in medical device software.",
        also_see=[]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="creepage",
        aliases=["creepage distance", "air clearance", "clearance", "insulation"],
        primary_standard="IEC_60601-1",
        primary_section="8.9",
        primary_note="Creepage distances and air clearances in Table 6 and Table 12.",
        also_see=[]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="usability",
        aliases=["usability engineering", "human factors", "use error"],
        primary_standard="IEC_60601-1",
        primary_section="12",
        primary_note="Accuracy of controls. Full usability in IEC 60601-1-6 / IEC 62366.",
        also_see=[
            {"standard": "ISO_14971", "section": "5", "note": "Use errors as hazards"},
        ]
    ))
    
    library.add_cross_reference(CrossReference(
        topic="alarm",
        aliases=["alarms", "alarm system", "alert", "alarm signal"],
        primary_standard="IEC_60601-1",
        primary_section="Annex A",
        primary_note="Normative annex for alarm systems. Full requirements in IEC 60601-1-8.",
        also_see=[]
    ))
    
    # Save the index
    library.save(get_index_path())
    logger.info(f"Created example index with {len(library.standards)} standards and {len(set(x.topic for x in library.cross_references.values()))} cross-reference topics")


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    Tool(
        name="list_available_standards",
        description="""List all regulatory standards available in the library.
        
Use this first to see what standards you have access to.
Returns the ID, title, and brief description of each standard.
""",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    
    Tool(
        name="lookup_topic",
        description="""Look up a topic directly in the cross-reference index.
        
This is the FASTEST way to find where a topic is covered. Returns the primary 
standard and section, plus other relevant locations.

USE THIS FIRST when you know what topic you're looking for. Only fall back to 
find_relevant_standards if the topic isn't in the cross-reference index.

Examples:
- "leakage current" → IEC 60601-1 Section 8.7, also see Annex F
- "software safety classification" → IEC 62304 Section 4.3
- "risk management" → ISO 14971 Section 4-10
- "EMC" → IEC 60601-1-2 (entire standard)
""",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to look up (e.g., 'leakage current', 'software classification', 'EMC')"
                }
            },
            "required": ["topic"]
        }
    ),
    
    Tool(
        name="find_relevant_standards",
        description="""Find which standard(s) are most relevant for a topic or question.
        
NOTE: Try lookup_topic FIRST - it's faster and more precise. Use this tool as a 
FALLBACK when the topic isn't in the cross-reference index.

This searches through all standards' metadata using keyword matching.
Returns ranked list of relevant standards with explanations of why they match.

Examples:
- "patient leakage current limits" → IEC 60601-1
- "software safety classification" → IEC 62304
- "implantable device requirements" → ISO 14708-1
- "risk analysis process" → ISO 14971
""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic, question, or requirement you're looking for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of standards to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    ),
    
    Tool(
        name="get_all_standards_for_semantic_search",
        description="""Get descriptions of ALL available standards so you can determine which is most relevant.
        
Use this when you need to find which standard covers a topic, and the topic might use 
different terminology than what's in the index. This returns full descriptions of all 
standards so YOU (Claude) can use your semantic understanding to find the right one.

For example, if someone asks about "creepage distances" or "dielectric strength", 
this will return the IEC 60601-1 description which mentions "electrical safety" - 
and you'll understand these are related.
""",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    
    Tool(
        name="get_standard_overview",
        description="""Get detailed information about a specific standard.
        
Returns the standard's scope, what topics it covers, its section structure,
and related standards. Use this to understand what's in a standard before
reading the full PDF.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID (e.g., 'IEC_60601-1', 'ISO_14971')"
                }
            },
            "required": ["standard_id"]
        }
    ),
    
    Tool(
        name="find_section",
        description="""Find which section of a standard covers a specific topic.
        
Use this to narrow down where to look within a standard.
Returns the relevant section number(s) and descriptions.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID"
                },
                "topic": {
                    "type": "string",
                    "description": "The topic you're looking for"
                }
            },
            "required": ["standard_id", "topic"]
        }
    ),
    
    Tool(
        name="get_related_standards",
        description="""Get standards that are related to a given standard.
        
Useful for understanding the regulatory ecosystem and finding additional
relevant requirements.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID"
                }
            },
            "required": ["standard_id"]
        }
    ),
    
    Tool(
        name="get_pdf_for_reading",
        description="""Get the PDF file for a standard so you can read it directly.
        
Use this when you need to read the actual standard content.
Returns the file path and basic info about the PDF.

IMPORTANT: After calling this, you can ask the user to share the PDF,
or if you have file access, you can read it directly.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID"
                }
            },
            "required": ["standard_id"]
        }
    ),
    
    Tool(
        name="find_table",
        description="""Find which table in a standard contains specific information.
        
Use this when looking for specific values, limits, or classifications that are
typically found in tables. Returns matching tables with their descriptions.

Examples:
- "leakage current limits" → Table 3 in IEC 60601-1
- "software safety classification" → Table in IEC 62304
- "applied part classification" → Table 1 in IEC 60601-1
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID"
                },
                "topic": {
                    "type": "string",
                    "description": "What information you're looking for"
                }
            },
            "required": ["standard_id", "topic"]
        }
    ),
    
    Tool(
        name="find_annex",
        description="""Find annexes in a standard that relate to a specific section or topic.
        
Use this when:
- You found a requirement and want supporting test methods or guidance
- You want to know what normative annexes apply to a section
- You're looking for informative rationale or examples

Returns matching annexes with their normative status and related sections.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID"
                },
                "section_or_topic": {
                    "type": "string",
                    "description": "Section number (e.g., '8.7') or topic (e.g., 'leakage current')"
                }
            },
            "required": ["standard_id", "section_or_topic"]
        }
    ),
    
    Tool(
        name="find_figure",
        description="""Find which figure in a standard illustrates specific information.
        
Use this when looking for diagrams, flowcharts, test circuits, or visual references.
Returns matching figures with their descriptions and locations.

Examples:
- "test circuit" → Figure F.1 in IEC 60601-1
- "risk management process" → Figure 1 in ISO 14971
- "software development" → Figure 1 in IEC 62304
""",
        inputSchema={
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "Standard ID"
                },
                "topic": {
                    "type": "string",
                    "description": "What you're looking for (e.g., 'test circuit', 'flowchart', 'classification')"
                }
            },
            "required": ["standard_id", "topic"]
        }
    ),
]


# =============================================================================
# Tool Handlers
# =============================================================================

async def handle_list_available_standards(arguments: dict) -> str:
    """List all available standards."""
    if not library.standards:
        return "No standards in library. Add PDFs to the data/pdfs directory and update the index."
    
    output = ["# Available Regulatory Standards\n"]
    
    for std_id, std in sorted(library.standards.items()):
        output.append(f"## {std.short_title}")
        output.append(f"**ID:** `{std_id}`")
        output.append(f"**Title:** {std.title}")
        output.append(f"**Organization:** {std.organization} ({std.year})")
        output.append(f"\n{std.description}\n")
        
        # Check if PDF exists
        pdf_path = library.get_pdf_path(std_id)
        if pdf_path:
            output.append(f"📄 PDF available: `{std.filename}`\n")
        else:
            output.append(f"⚠️ PDF not found: `{std.filename}` (add to {library.pdf_directory})\n")
        
        output.append("---\n")
    
    # Show cross-reference stats
    unique_topics = len(set(xref.topic for xref in library.cross_references.values()))
    if unique_topics > 0:
        output.append(f"\n📚 **Cross-reference index:** {unique_topics} topics indexed for quick lookup")
        output.append("Use `lookup_topic` for fastest access to specific topics.")
    
    return "\n".join(output)


async def handle_lookup_topic(arguments: dict) -> str:
    """Look up a topic in the cross-reference index."""
    topic = arguments["topic"]
    
    xref = library.lookup_topic(topic)
    
    if not xref:
        return f"""Topic '{topic}' not found in cross-reference index.

**Options:**
1. Try `find_relevant_standards` for keyword search across all standards
2. Use `get_all_standards_for_semantic_search` to browse all standards
3. The topic might be indexed under a different name - try synonyms

**Tip:** Common indexed topics include: leakage current, software safety classification, 
risk management, EMC, biocompatibility, essential performance, applied parts"""
    
    output = [f"# Topic: {xref.topic}\n"]
    
    # Show aliases if any
    if xref.aliases:
        output.append(f"**Also known as:** {', '.join(xref.aliases)}\n")
    
    # Primary location
    std = library.standards.get(xref.primary_standard)
    std_name = std.short_title if std else xref.primary_standard
    
    output.append("## Primary Location\n")
    output.append(f"**{std_name}** — Section {xref.primary_section}")
    if xref.primary_note:
        output.append(f"\n{xref.primary_note}")
    output.append("")
    
    # Also see
    if xref.also_see:
        output.append("\n## Also See\n")
        for ref in xref.also_see:
            ref_std = library.standards.get(ref['standard'])
            ref_name = ref_std.short_title if ref_std else ref['standard']
            line = f"- **{ref_name}** — Section {ref['section']}"
            if ref.get('note'):
                line += f" ({ref['note']})"
            output.append(line)
    
    output.append(f"\n\n💡 Use `get_pdf_for_reading` with `{xref.primary_standard}` to read the primary source.")
    
    return "\n".join(output)


async def handle_find_relevant_standards(arguments: dict) -> str:
    """Find standards relevant to a query."""
    query = arguments["query"]
    limit = arguments.get("limit", 3)
    
    results = library.find_standards(query, limit)
    
    if not results:
        # No keyword matches - suggest using semantic search
        return f"""No direct keyword matches for '{query}'. 

This might be because the search terms differ from how the standards are indexed.

**Suggestion:** Use `get_all_standards_for_semantic_search` to see all available standards 
and their descriptions - you can then determine which is most relevant based on your 
understanding of the topic."""
    
    output = [f"# Standards Relevant to: \"{query}\"\n"]
    
    for i, (std, score) in enumerate(results, 1):
        output.append(f"## {i}. {std.short_title} (relevance: {score:.1f})")
        output.append(f"**ID:** `{std.id}`")
        output.append(f"\n{std.description}\n")
        
        # Show which topics matched
        matching_topics = [t for t in std.key_topics if query.lower() in t.lower() or t.lower() in query.lower()]
        if matching_topics:
            output.append(f"**Matching topics:** {', '.join(matching_topics[:5])}\n")
        
        # Show relevant sections
        matching_sections = {k: v for k, v in std.sections.items() if query.lower() in v.lower()}
        if matching_sections:
            output.append("**Relevant sections:**")
            for sec, desc in list(matching_sections.items())[:3]:
                output.append(f"- Section {sec}: {desc}")
            output.append("")
        
        output.append("---\n")
    
    output.append("\n💡 **Tip:** Use `get_pdf_for_reading` with the standard ID to access the full document.")
    
    return "\n".join(output)


async def handle_get_all_standards_semantic(arguments: dict) -> str:
    """Return all standards with full descriptions for Claude to evaluate semantically."""
    if not library.standards:
        return "No standards in library."
    
    output = ["# All Available Standards\n"]
    output.append("Review these descriptions to find which standard(s) are most relevant to your query.\n")
    output.append("---\n")
    
    for std_id, std in sorted(library.standards.items()):
        output.append(f"## {std.short_title} (`{std.id}`)")
        output.append(f"\n**Title:** {std.title}\n")
        output.append(f"**Scope:** {std.scope}\n")
        output.append(f"**Description:** {std.description}\n")
        
        output.append("**Sections:**")
        for sec, desc in std.sections.items():
            output.append(f"- {sec}: {desc}")
        
        # Show annexes with related sections
        if std.annexes:
            output.append("\n**Annexes:**")
            for annex_id, annex_data in std.annexes.items():
                status = "normative" if annex_data['normative'] else "informative"
                related = annex_data.get('related_sections', [])
                line = f"- {annex_id} ({status}): {annex_data['description']}"
                if related:
                    line += f" [relates to: {', '.join(related)}]"
                output.append(line)
        
        # Show key tables with locations
        if std.key_tables:
            output.append("\n**Key Tables:**")
            for table_id, table_data in std.key_tables.items():
                location = table_data.get('location', '')
                related = table_data.get('related_sections', [])
                if location:
                    line = f"- {table_id} (Section {location}): {table_data['description']}"
                else:
                    line = f"- {table_id}: {table_data['description']}"
                if related:
                    line += f" [also: {', '.join(related)}]"
                output.append(line)
        
        output.append(f"\n**Key topics:** {', '.join(std.key_topics)}")
        
        if std.key_terms:
            output.append(f"\n**Defined terms:** {', '.join(std.key_terms[:10])}")
            if len(std.key_terms) > 10:
                output.append(f"  ...and {len(std.key_terms) - 10} more")
        
        pdf_path = library.get_pdf_path(std_id)
        output.append(f"\n**PDF:** {'✓ Available' if pdf_path else '✗ Missing'}")
        output.append("\n---\n")
    
    output.append("\nOnce you identify the relevant standard(s), use `get_pdf_for_reading` to access the full document.")
    
    return "\n".join(output)


async def handle_get_standard_overview(arguments: dict) -> str:
    """Get detailed overview of a standard."""
    standard_id = arguments["standard_id"]
    
    std = library.standards.get(standard_id)
    if not std:
        available = ", ".join(library.standards.keys())
        return f"Standard '{standard_id}' not found. Available standards: {available}"
    
    output = [f"# {std.short_title}"]
    output.append(f"**Full Title:** {std.title}")
    output.append(f"**Organization:** {std.organization}")
    output.append(f"**Version/Year:** {std.year}")
    output.append(f"**Pages:** ~{std.pages}")
    output.append("")
    
    output.append("## Scope")
    output.append(std.scope)
    output.append("")
    
    output.append("## Description")
    output.append(std.description)
    output.append("")
    
    output.append("## Sections")
    for sec, desc in std.sections.items():
        output.append(f"- **{sec}:** {desc}")
    output.append("")
    
    # Annexes
    if std.annexes:
        output.append("## Annexes")
        for annex_id, annex_data in std.annexes.items():
            status = "(normative)" if annex_data['normative'] else "(informative)"
            related = annex_data.get('related_sections', [])
            line = f"- **{annex_id}** {status}: {annex_data['description']}"
            if related:
                line += f" [relates to: {', '.join(related)}]"
            output.append(line)
        output.append("")
    
    # Key Tables
    if std.key_tables:
        output.append("## Key Tables")
        for table_id, table_data in std.key_tables.items():
            location = table_data.get('location', '')
            related = table_data.get('related_sections', [])
            if location:
                line = f"- **{table_id}** (Section {location}): {table_data['description']}"
            else:
                line = f"- **{table_id}:** {table_data['description']}"
            if related:
                line += f" [also: {', '.join(related)}]"
            output.append(line)
        output.append("")
    
    # Key Figures
    if std.key_figures:
        output.append("## Key Figures")
        for figure_id, figure_data in std.key_figures.items():
            location = figure_data.get('location', '')
            related = figure_data.get('related_sections', [])
            if location:
                line = f"- **{figure_id}** (Section {location}): {figure_data['description']}"
            else:
                line = f"- **{figure_id}:** {figure_data['description']}"
            if related:
                line += f" [also: {', '.join(related)}]"
            output.append(line)
        output.append("")
    
    # Key Terms
    if std.key_terms:
        output.append("## Key Defined Terms")
        terms_display = std.key_terms[:15]
        output.append(", ".join(terms_display))
        if len(std.key_terms) > 15:
            output.append(f"... and {len(std.key_terms) - 15} more terms")
        output.append("")
    
    output.append("## Key Topics (Searchable)")
    topics = std.key_topics
    for i in range(0, len(topics), 4):
        output.append("- " + ", ".join(topics[i:i+4]))
    output.append("")
    
    output.append("## Related Standards")
    for rel in std.related_standards:
        rel_id = rel['id']
        relationship = rel.get('relationship', 'related')
        description = rel.get('description', '')
        output.append(f"- **{rel_id}** ({relationship})")
        if description:
            output.append(f"  {description}")
    output.append("")
    
    # Notes
    if std.notes:
        output.append("## Notes")
        output.append(std.notes)
        output.append("")
    
    # PDF status
    pdf_path = library.get_pdf_path(standard_id)
    if pdf_path:
        output.append(f"📄 **PDF Available:** `{pdf_path}`")
    else:
        output.append(f"⚠️ **PDF Not Found:** Add `{std.filename}` to `{library.pdf_directory}/`")
    
    return "\n".join(output)


async def handle_find_section(arguments: dict) -> str:
    """Find relevant section in a standard."""
    standard_id = arguments["standard_id"]
    topic = arguments["topic"]
    
    std = library.standards.get(standard_id)
    if not std:
        return f"Standard '{standard_id}' not found."
    
    topic_lower = topic.lower()
    
    # Search sections
    matching_sections = []
    for sec, desc in std.sections.items():
        if topic_lower in desc.lower():
            matching_sections.append((sec, desc, "title match"))
    
    # Search key topics to infer sections
    for key_topic in std.key_topics:
        if topic_lower in key_topic.lower() or key_topic.lower() in topic_lower:
            # Try to map topic to section (heuristic)
            for sec, desc in std.sections.items():
                if key_topic.lower() in desc.lower():
                    if (sec, desc, "topic match") not in matching_sections:
                        matching_sections.append((sec, desc, "topic match"))
    
    if not matching_sections:
        output = [f"# Section Search: \"{topic}\" in {std.short_title}\n"]
        output.append(f"No exact section match found for '{topic}'.\n")
        output.append("**Available sections:**")
        for sec, desc in std.sections.items():
            output.append(f"- {sec}: {desc}")
        output.append("\n💡 Try reading the full standard or searching with different keywords.")
        return "\n".join(output)
    
    output = [f"# Sections for \"{topic}\" in {std.short_title}\n"]
    output.append("**Likely relevant sections:**\n")
    
    for sec, desc, match_type in matching_sections[:5]:
        output.append(f"- **Section {sec}:** {desc}")
    
    output.append(f"\n💡 Use `get_pdf_for_reading` with `{standard_id}` to read these sections.")
    
    return "\n".join(output)


async def handle_find_table(arguments: dict) -> str:
    """Find tables in a standard that match a topic or section."""
    standard_id = arguments["standard_id"]
    topic = arguments["topic"]
    
    std = library.standards.get(standard_id)
    if not std:
        return f"Standard '{standard_id}' not found."
    
    topic_lower = topic.lower()
    
    if not std.key_tables:
        output = [f"# Table Search: \"{topic}\" in {std.short_title}\n"]
        output.append("⚠️ No tables have been indexed for this standard.\n")
        output.append("The standard may still contain relevant tables - use `get_pdf_for_reading` to check the document directly.")
        return "\n".join(output)
    
    # Search key_tables
    matching_tables = []
    for table_id, table_data in std.key_tables.items():
        desc = table_data['description']
        location = table_data.get('location', '')
        related = table_data.get('related_sections', [])
        
        # Check if query matches description
        if topic_lower in desc.lower():
            matching_tables.append((table_id, desc, location, related, "description match"))
            continue
        
        # Check if query matches location or related sections (for section-based searches)
        section_match = False
        if location:
            if topic_lower == location.lower() or topic_lower.startswith(location.lower() + ".") or location.lower().startswith(topic_lower + "."):
                section_match = True
        for rel_sec in related:
            if topic_lower == rel_sec.lower() or topic_lower.startswith(rel_sec.lower() + ".") or rel_sec.lower().startswith(topic_lower + "."):
                section_match = True
                break
        
        if section_match:
            matching_tables.append((table_id, desc, location, related, "section match"))
            continue
        
        # Check individual words in description
        for word in topic_lower.split():
            if len(word) > 3 and word in desc.lower():
                matching_tables.append((table_id, desc, location, related, "word match"))
                break
    
    # Also search sections for table references
    for sec, desc in std.sections.items():
        if topic_lower in desc.lower() and "table" in desc.lower():
            matching_tables.append((f"See Section {sec}", desc, sec, [], "section reference"))
    
    if not matching_tables:
        output = [f"# Table Search: \"{topic}\" in {std.short_title}\n"]
        output.append(f"No tables directly matching '{topic}' found.\n")
        output.append("**Available tables in this standard:**")
        for table_id, table_data in std.key_tables.items():
            location = table_data.get('location', '')
            related = table_data.get('related_sections', [])
            if location:
                line = f"- **{table_id}** (Section {location}): {table_data['description']}"
            else:
                line = f"- **{table_id}:** {table_data['description']}"
            if related:
                line += f" [also: {', '.join(related)}]"
            output.append(line)
        output.append(f"\n💡 Use `get_pdf_for_reading` to check the full document.")
        return "\n".join(output)
    
    output = [f"# Tables for \"{topic}\" in {std.short_title}\n"]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for item in matching_tables:
        if item[0] not in seen:
            seen.add(item[0])
            unique_matches.append(item)
    
    output.append("**Likely relevant tables:**\n")
    for table_id, desc, location, related, match_type in unique_matches[:5]:
        if location:
            line = f"- **{table_id}** (Section {location}): {desc}"
        else:
            line = f"- **{table_id}:** {desc}"
        if related:
            line += f"\n  Also referenced in: {', '.join(related)}"
        output.append(line)
    
    output.append(f"\n💡 Use `get_pdf_for_reading` with `{standard_id}` to view these tables.")
    
    return "\n".join(output)


async def handle_find_figure(arguments: dict) -> str:
    """Find figures in a standard that match a topic or section."""
    standard_id = arguments["standard_id"]
    topic = arguments["topic"]
    
    std = library.standards.get(standard_id)
    if not std:
        return f"Standard '{standard_id}' not found."
    
    topic_lower = topic.lower()
    
    if not std.key_figures:
        output = [f"# Figure Search: \"{topic}\" in {std.short_title}\n"]
        output.append("⚠️ No figures have been indexed for this standard.\n")
        output.append("The standard may still contain relevant figures - use `get_pdf_for_reading` to check the document directly.")
        return "\n".join(output)
    
    # Search key_figures
    matching_figures = []
    for figure_id, figure_data in std.key_figures.items():
        desc = figure_data['description']
        location = figure_data.get('location', '')
        related = figure_data.get('related_sections', [])
        
        # Check if query matches description
        if topic_lower in desc.lower():
            matching_figures.append((figure_id, desc, location, related, "description match"))
            continue
        
        # Check if query matches location or related sections (for section-based searches)
        section_match = False
        if location:
            if topic_lower == location.lower() or topic_lower.startswith(location.lower() + ".") or location.lower().startswith(topic_lower + "."):
                section_match = True
        for rel_sec in related:
            if topic_lower == rel_sec.lower() or topic_lower.startswith(rel_sec.lower() + ".") or rel_sec.lower().startswith(topic_lower + "."):
                section_match = True
                break
        
        if section_match:
            matching_figures.append((figure_id, desc, location, related, "section match"))
            continue
        
        # Check individual words in description
        for word in topic_lower.split():
            if len(word) > 3 and word in desc.lower():
                matching_figures.append((figure_id, desc, location, related, "word match"))
                break
    
    if not matching_figures:
        output = [f"# Figure Search: \"{topic}\" in {std.short_title}\n"]
        output.append(f"No figures directly matching '{topic}' found.\n")
        output.append("**Available figures in this standard:**")
        for figure_id, figure_data in std.key_figures.items():
            location = figure_data.get('location', '')
            related = figure_data.get('related_sections', [])
            if location:
                line = f"- **{figure_id}** (Section {location}): {figure_data['description']}"
            else:
                line = f"- **{figure_id}:** {figure_data['description']}"
            if related:
                line += f" [also: {', '.join(related)}]"
            output.append(line)
        output.append(f"\n💡 Use `get_pdf_for_reading` to check the full document.")
        return "\n".join(output)
    
    output = [f"# Figures for \"{topic}\" in {std.short_title}\n"]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for item in matching_figures:
        if item[0] not in seen:
            seen.add(item[0])
            unique_matches.append(item)
    
    output.append("**Likely relevant figures:**\n")
    for figure_id, desc, location, related, match_type in unique_matches[:5]:
        if location:
            line = f"- **{figure_id}** (Section {location}): {desc}"
        else:
            line = f"- **{figure_id}:** {desc}"
        if related:
            line += f"\n  Also referenced in: {', '.join(related)}"
        output.append(line)
    
    output.append(f"\n💡 Use `get_pdf_for_reading` with `{standard_id}` to view these figures.")
    
    return "\n".join(output)


async def handle_find_annex(arguments: dict) -> str:
    """Find annexes in a standard that relate to a section or topic."""
    standard_id = arguments["standard_id"]
    section_or_topic = arguments["section_or_topic"]
    
    std = library.standards.get(standard_id)
    if not std:
        return f"Standard '{standard_id}' not found."
    
    query_lower = section_or_topic.lower()
    
    if not std.annexes:
        output = [f"# Annex Search: \"{section_or_topic}\" in {std.short_title}\n"]
        output.append("⚠️ No annexes have been indexed for this standard.\n")
        output.append("The standard may still contain annexes - use `get_pdf_for_reading` to check the document directly.")
        return "\n".join(output)
    
    matching_annexes = []
    
    for annex_id, annex_data in std.annexes.items():
        desc = annex_data['description']
        normative = annex_data['normative']
        related_sections = annex_data.get('related_sections', [])
        
        # Check if the query matches a related section
        section_match = False
        for rel_sec in related_sections:
            # Match "8" to "8", "8.7", "8.7.3" etc.
            if query_lower == rel_sec.lower() or query_lower.startswith(rel_sec.lower() + ".") or rel_sec.lower().startswith(query_lower + "."):
                section_match = True
                break
            # Also check if rel_sec is "general" and query matches description
            if rel_sec.lower() == "general" and query_lower in desc.lower():
                section_match = True
                break
        
        # Check if query matches description
        desc_match = query_lower in desc.lower()
        
        if section_match or desc_match:
            match_type = "section" if section_match else "topic"
            matching_annexes.append((annex_id, desc, normative, related_sections, match_type))
    
    if not matching_annexes:
        output = [f"# Annex Search: \"{section_or_topic}\" in {std.short_title}\n"]
        output.append(f"No annexes directly related to '{section_or_topic}' found.\n")
        output.append("**Available annexes in this standard:**")
        for annex_id, annex_data in std.annexes.items():
            status = "(normative)" if annex_data['normative'] else "(informative)"
            related = annex_data.get('related_sections', [])
            line = f"- **{annex_id}** {status}: {annex_data['description']}"
            if related:
                line += f" [sections: {', '.join(related)}]"
            output.append(line)
        return "\n".join(output)
    
    output = [f"# Annexes for \"{section_or_topic}\" in {std.short_title}\n"]
    
    # Separate normative and informative
    normative_annexes = [a for a in matching_annexes if a[2] is True]
    informative_annexes = [a for a in matching_annexes if a[2] is False]
    
    if normative_annexes:
        output.append("**Normative Annexes (required for compliance):**\n")
        for annex_id, desc, _, related, match_type in normative_annexes:
            line = f"- **{annex_id}**: {desc}"
            if related:
                line += f"\n  Related sections: {', '.join(related)}"
            output.append(line)
        output.append("")
    
    if informative_annexes:
        output.append("**Informative Annexes (guidance/rationale):**\n")
        for annex_id, desc, _, related, match_type in informative_annexes:
            line = f"- **{annex_id}**: {desc}"
            if related:
                line += f"\n  Related sections: {', '.join(related)}"
            output.append(line)
        output.append("")
    
    output.append(f"💡 Use `get_pdf_for_reading` with `{standard_id}` to view these annexes.")
    
    return "\n".join(output)


async def handle_get_related_standards(arguments: dict) -> str:
    """Get related standards."""
    standard_id = arguments["standard_id"]
    
    std = library.standards.get(standard_id)
    if not std:
        return f"Standard '{standard_id}' not found."
    
    output = [f"# Standards Related to {std.short_title}\n"]
    
    if not std.related_standards:
        output.append("No related standards listed.")
        return "\n".join(output)
    
    for rel_id in std.related_standards:
        rel_std = library.standards.get(rel_id)
        if rel_std:
            output.append(f"## {rel_std.short_title}")
            output.append(f"**ID:** `{rel_id}`")
            output.append(f"**Title:** {rel_std.title}")
            output.append(f"\n{rel_std.description}\n")
            
            pdf_path = library.get_pdf_path(rel_id)
            if pdf_path:
                output.append(f"📄 PDF available\n")
            output.append("---\n")
        else:
            output.append(f"## {rel_id}")
            output.append("(Not in library - you may need to obtain this standard)\n")
            output.append("---\n")
    
    return "\n".join(output)


async def handle_get_pdf_for_reading(arguments: dict) -> str:
    """Get PDF path for reading."""
    standard_id = arguments["standard_id"]
    
    std = library.standards.get(standard_id)
    if not std:
        return f"Standard '{standard_id}' not found."
    
    pdf_path = library.get_pdf_path(standard_id)
    
    output = [f"# PDF Access: {std.short_title}\n"]
    
    if pdf_path:
        output.append(f"**File:** `{pdf_path}`")
        output.append(f"**Size:** ~{std.pages} pages")
        output.append(f"\n**Full path:** `{pdf_path.absolute()}`")
        output.append("")
        output.append("## How to Read This Standard")
        output.append("")
        output.append("**Option 1 - Claude Desktop:** Ask me to read the PDF by sharing it in the conversation.")
        output.append("")
        output.append("**Option 2 - API:** Include the PDF content in your API request.")
        output.append("")
        output.append("**Option 3 - Direct:** Open the file at the path above.")
        output.append("")
        output.append("## Quick Reference")
        output.append(f"\n{std.description}")
        output.append("\n**Sections:**")
        for sec, desc in list(std.sections.items())[:5]:
            output.append(f"- {sec}: {desc}")
        if len(std.sections) > 5:
            output.append(f"- ... and {len(std.sections) - 5} more sections")
    else:
        output.append(f"⚠️ **PDF Not Found**")
        output.append(f"\nExpected file: `{library.pdf_directory}/{std.filename}`")
        output.append(f"\nPlease add the PDF file to this location and restart the server.")
    
    return "\n".join(output)


# Tool handler dispatcher
TOOL_HANDLERS = {
    "list_available_standards": handle_list_available_standards,
    "lookup_topic": handle_lookup_topic,
    "find_relevant_standards": handle_find_relevant_standards,
    "get_all_standards_for_semantic_search": handle_get_all_standards_semantic,
    "get_standard_overview": handle_get_standard_overview,
    "find_section": handle_find_section,
    "find_table": handle_find_table,
    "find_figure": handle_find_figure,
    "find_annex": handle_find_annex,
    "get_related_standards": handle_get_related_standards,
    "get_pdf_for_reading": handle_get_pdf_for_reading,
}


# =============================================================================
# MCP Resource Handlers (for direct PDF access)
# =============================================================================

@server.list_resources()
async def list_resources() -> ListResourcesResult:
    """List PDFs as resources."""
    resources = []
    
    for std_id, std in library.standards.items():
        pdf_path = library.get_pdf_path(std_id)
        if pdf_path:
            resources.append(Resource(
                uri=f"standards://{std_id}/pdf",
                name=f"{std.short_title} (PDF)",
                description=std.title,
                mimeType="application/pdf"
            ))
    
    return ListResourcesResult(resources=resources)


@server.read_resource()
async def read_resource(request: ReadResourceRequest) -> ReadResourceResult:
    """Read a PDF resource."""
    uri = request.params.uri
    
    # Parse URI: standards://IEC_60601-1/pdf
    if uri.startswith("standards://") and uri.endswith("/pdf"):
        std_id = uri[12:-4]  # Extract standard ID
        
        pdf_path = library.get_pdf_path(std_id)
        if pdf_path and pdf_path.exists():
            # Read and base64 encode the PDF
            with open(pdf_path, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")
            
            return ReadResourceResult(
                contents=[BlobResourceContents(
                    uri=uri,
                    mimeType="application/pdf",
                    blob=content
                )]
            )
    
    return ReadResourceResult(contents=[
        TextResourceContents(
            uri=uri,
            mimeType="text/plain",
            text=f"Resource not found: {uri}"
        )
    ])


# =============================================================================
# MCP Protocol Handlers
# =============================================================================

@server.list_tools()
async def list_tools() -> ListToolsResult:
    """Return list of available tools."""
    return ListToolsResult(tools=TOOLS)


@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls."""
    tool_name = request.params.name
    arguments = request.params.arguments or {}
    
    logger.info(f"Tool call: {tool_name} with args: {arguments}")
    
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")]
        )
    
    try:
        result = await handler(arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=result)]
        )
    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Run the MCP server."""
    initialize()
    
    logger.info("Starting Standards Librarian MCP server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
