from pr_agent.diff_parser import parse_hunks
from pr_agent.validator import validate_findings
from pr_agent.models import Finding

def main()->None:
    patch = '''@@ -33,7 +33,7 @@ def main() -> None:
 
     print("1. Extracting text from PDFs...")
     for pdf_path in input_pdfs:
-        text_path = extract_text_from_pdf(pdf_path)
+        text_path: Path = extract_text_from_pdf(pdf_path)
         extracted_text_paths.append(text_path)
 
     extracted_facts_path = output_dir / "extracted_facts.json"
    '''
    hunks = parse_hunks(filename='main.py',patch=patch)
    findings = [
            Finding(
                filename='main.py',
                line=36,
                severity= "error",
                category= 'type-error',
                message='The annotation does not match the function return type',
                ),
            Finding(
                filename='main.py',
                line=999,
                severity='warning',
                category="bug",
                message="This finding points to a nonexistent changed line",
                ),
            ]
    validated_findings = validate_findings(findings=findings, hunks = hunks)
    print("Parsed Hunks:")
    for hunk in hunks:
        print(f"-{hunk.filename}"
              f"added = {sorted(hunk.added_lines)}"
              f"deleted={sorted(hunk.deleted_lines)}"
              )

    print("\nProposed findings:")
    for finding in findings:
        print(f"- {finding.filename}:{finding.line} -- {finding.message}")

    print("\nValidated findings:")
    for finding in validated_findings:
        print(f"- {finding.filename}: {finding.line} --{finding.message}")

    assert len(validated_findings)==1
    assert validated_findings[0].line==36

    print("\nValidation passes.")

if __name__=="__main__":
    main()
