
#import "@preview/kroki:0.1.0": kroki

#set page(paper: "a4", margin: 2cm)

= Test Diagram

#kroki(
  format: "svg",
  target: "mermaid",
  "graph TD; A[Input] --> B[Output];"
)
