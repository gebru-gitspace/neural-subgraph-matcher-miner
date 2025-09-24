import os
import json

class HTMLTemplateProcessor:
    def __init__(self, template_path: str = None):
        # You can expand this to load a real HTML template file if needed
        self.template_path = template_path

    def generate_filename(self, graph_data: dict, base_name: str) -> str:
        return f"{base_name}_graph.html"

    def process_template(self, graph_data: dict, output_filename: str, output_dir: str = "./plots") -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # Simple interactive vis.js HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{graph_data['metadata'].get('title', 'Graph Visualization')}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
<style>
#network {{
  width: 100%;
  height: 600px;
  border: 1px solid lightgray;
}}
</style>
</head>
<body>
<h3>{graph_data['metadata'].get('title', 'Graph Visualization')}</h3>
<div id="network"></div>
<script>
var nodes = new vis.DataSet({json.dumps(graph_data['nodes'])});
var edges = new vis.DataSet({json.dumps(graph_data['edges'])});
var container = document.getElementById('network');
var data = {{ nodes: nodes, edges: edges }};
var options = {{
    nodes: {{ shape: 'dot', size: 16 }},
    edges: {{ arrows: {{ to: {{ enabled: {str(graph_data['metadata'].get('isDirected', False)).lower()} }} }} }},
    physics: {{ stabilization: false }}
}};
var network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path


def validate_graph_data(graph_data: dict) -> bool:
    """Basic validation for nodes and edges."""
    if not isinstance(graph_data, dict):
        return False
    if "nodes" not in graph_data or "edges" not in graph_data:
        return False
    if not isinstance(graph_data["nodes"], list) or not isinstance(graph_data["edges"], list):
        return False
    return True
