# RAG System for Document Processing and Question Answering

## 📚 Overview
This is a Retrieval-Augmented Generation (RAG) system that helps you:
- Process documents (PDFs, web pages)
- Search through them intelligently
- Get relevant answers to your questions

Think of it as your personal AI research assistant!

## 🎯 Key Features
- 📄 Document Processing: Handle PDFs and web content
- 🔍 Smart Search: Find relevant information quickly
- 🧠 AI-Powered: Uses embeddings to understand content
- 💾 Persistent Storage: Save and reload your processed documents
- ⚡ Multiple Search Methods: Regular and keyword-based search

## 🏗️ System Architecture
The system is organized into several key components:

1. **Ingestion**:
   - 📄 Document Processing: Convert raw text into Document objects
   - 🔍 Embedding: Convert text into numerical vectors
   - 💾 Indexing: Save and load searchable indexes

graph TD
    A[Ingestion] --> B[Document Processing]
    B --> C[Embedding]
    C --> D[Indexing]
    D --> E[Search]
    E --> F[Answer]

Each component has a specific role:
1. **Content Handler**: Gets text from PDFs and websites
2. **Document Processor**: Prepares documents for AI processing
3. **Embedding Manager**: Adds AI understanding to documents
4. **Index Manager**: Organizes documents for quick search
5. **Query Processor**: Handles your questions
6. **RAG Pipeline**: Coordinates the search process

## 🚀 Getting Started

### Prerequisites

