# Challenge 1B: Multi-Collection PDF Analysis

Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases.

## 🎯 Overview

This solution extends basic PDF processing to provide **persona-based content analysis**. It intelligently identifies and ranks content sections based on their relevance to specific user roles and tasks.

## 📂 Project Structure

```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides (7 documents)
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning  
│   ├── PDFs/                       # Acrobat tutorials (15 documents)
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides (9 documents)
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── challenge1b_processor.py        # Main processing script
└── README.md                       # This file
```

## 🎭 Collections & Personas

### Collection 1: Travel Planning 🌍
- **Challenge ID**: round_1b_002
- **Persona**: Travel Planner
- **Task**: Plan a 4-day trip for 10 college friends to South of France
- **Documents**: 7 travel guides covering accommodation, transportation, activities, budget

### Collection 2: Adobe Acrobat Learning 📋
- **Challenge ID**: round_1b_003  
- **Persona**: HR Professional
- **Task**: Create and manage fillable forms for onboarding and compliance
- **Documents**: 15 Acrobat guides covering forms, workflows, security, templates

### Collection 3: Recipe Collection 🍽️
- **Challenge ID**: round_1b_001
- **Persona**: Food Contractor
- **Task**: Prepare vegetarian buffet-style dinner menu for corporate gathering
- **Documents**: 9 cooking guides covering vegetarian cuisine, catering, food safety

## 🧠 Key Features

### Persona-Based Analysis
- **Context-Aware Processing**: Understands different user roles and priorities
- **Keyword Relevance Scoring**: Uses weighted keyword matching for each persona
- **Task-Specific Content Extraction**: Filters content based on specific job requirements

### Advanced Content Analysis
- **Importance Ranking**: Ranks sections by relevance to persona and task
- **Multi-Level Heading Detection**: Identifies H1, H2, H3 headings using font and content analysis
- **Content Summarization**: Provides refined text excerpts for key sections

### Intelligent Processing
- **Multi-Collection Support**: Processes different document types simultaneously
- **Scalable Architecture**: Handles varying document counts per collection
- **Error-Resilient**: Continues processing even if some documents are missing

## 📊 Output Format

### Input JSON Structure
```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [
    {"filename": "doc.pdf", "title": "Document Title"}
  ],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Task description"}
}
```

### Output JSON Structure
```json
{
  "metadata": {
    "input_documents": ["list of processed files"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "total_sections_found": 150,
    "processing_timestamp": "2024-01-20 15:30:45"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Section Title",
      "importance_rank": 1,
      "page_number": 5,
      "relevance_score": 0.95
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf", 
      "refined_text": "Key content excerpt...",
      "page_number": 5,
      "relevance_score": 0.92
    }
  ]
}
```

## 🚀 How to Run

### Prerequisites
Make sure you have the required dependencies from your existing pdf-processor:
```bash
pip install PyPDF2==3.0.1 pdfplumber==0.10.3
```

### Execute Analysis
```bash
cd C:\Users\nikhil\Challenge_1b
python challenge1b_processor.py
```

### Expected Behavior
- Processes all three collections automatically
- Generates `challenge1b_output.json` for each collection
- Provides persona-specific relevance scoring
- Ranks content by importance for each use case
- Handles missing PDFs gracefully with sample data

## 🎯 Persona-Specific Keywords

### Travel Planner (High Priority)
- accommodation, hotel, booking, transportation, flight, train
- itinerary, schedule, planning, budget, cost, activities
- group, friends, college, student, discount

### HR Professional (High Priority)  
- onboarding, compliance, legal, regulation, policy
- form, document, template, fillable, workflow
- signature, authentication, verification, approval

### Food Contractor (High Priority)
- vegetarian, vegan, buffet, catering, corporate
- menu, recipe, quantity, portion, planning
- dietary, restriction, allergy, gluten-free

## 📈 Performance

- **Processing Speed**: Optimized for large document collections
- **Relevance Accuracy**: Context-aware scoring for precise content ranking  
- **Scalability**: Handles multiple collections simultaneously
- **Memory Efficient**: Processes documents incrementally

## 🔧 Technical Implementation

### Core Components
1. **PersonaBasedPDFAnalyzer**: Main analysis engine
2. **Keyword Matching System**: Weighted relevance calculation
3. **Section Extraction**: Advanced heading detection
4. **Content Ranking**: Importance scoring algorithm

### Advanced Features
- **Multi-criteria heading detection** (font, content, position analysis)
- **Persona-specific keyword weighting** (high/medium/low priority)
- **Content relevance scoring** with normalized metrics
- **Importance ranking** based on task alignment

## 📋 Sample Output

The processor generates realistic sample data when PDFs are not available, demonstrating:
- **Travel Planner**: Accommodation options, transportation guides, budget planning
- **HR Professional**: Form creation, compliance requirements, workflow automation
- **Food Contractor**: Menu planning, catering guidelines, dietary restrictions

## 🏆 Challenge Compliance

This solution fully addresses Challenge 1B requirements:
- ✅ Multi-collection document processing
- ✅ Persona-based content analysis  
- ✅ Importance ranking of sections
- ✅ Structured JSON input/output
- ✅ Advanced content extraction with relevance scoring
