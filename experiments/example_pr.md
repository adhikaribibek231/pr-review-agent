PR #1: Improve parser and error handling
State: open
Base branch: main
Source branch: feature/parser-fixes

================================================================================
File: main.py
Status: modified
Additions: 2
Deletions: 2
Patch:
@@ -33,7 +33,7 @@ def main() -> None:

     print("1. Extracting text from PDFs...")
     for pdf_path in input_pdfs:
-        text_path = extract_text_from_pdf(pdf_path)
+        text_path: Path = extract_text_from_pdf(pdf_path)
         extracted_text_paths.append(text_path)

     extracted_facts_path = output_dir / "extracted_facts.json"
@@ -72,5 +72,5 @@ def main() -> None:
     results = process_facts(extracted_facts_path)

-    print(results)
+    print(f"Processed {len(results)} results")

     save_results(results)

================================================================================
File: src/parser.py
Status: modified
Additions: 3
Deletions: 1
Patch:
@@ -18,6 +18,8 @@ def parse_document(path: Path) -> str:
     if not path.exists():
-        return ""
+        raise FileNotFoundError(f"Document not found: {path}")
+
+    encoding = "utf-8"

-    return path.read_text()
+    return path.read_text(encoding=encoding)

================================================================================
File: README.md
Status: modified
Additions: 1
Deletions: 0
Patch:
@@ -40,3 +40,4 @@ Run the application with:
 uv run python main.py
+
+Python 3.12 or newer is required.
