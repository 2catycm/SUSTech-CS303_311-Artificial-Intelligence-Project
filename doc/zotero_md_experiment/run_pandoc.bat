rem pandoc --filter pandoc-citeproc "zotero基础使用.md" --bibliography ./我的文库.bib -o "report.pdf"

rem pandoc --citeproc "zotero_basic_usage.md" --bibliography ./我的文库.bib -o "report.docx"

pandoc --citeproc "zotero基础使用.md" --bibliography ./我的文库.bib -o "report.tex"

rem pandoc  --lua-filter=zotero.lua  "zotero基础使用.md" --bibliography ./我的文库.bib -o "report.pdf"

pause