"""Execute trusted overview cells and README examples using ordinary Python.

No Jupyter-kernel or nbformat-package validation is claimed. This checker only
supports this repository's plain Python cells, explicit print and plt.show.
Run after installing research dependencies: python -m experiments.check_presentation
"""
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    path=root/'notebooks/project_overview.ipynb'
    notebook=json.loads(path.read_text())
    if notebook['nbformat']!=4 or len({c['id'] for c in notebook['cells']})!=len(notebook['cells']):
        raise ValueError('invalid notebook version or nonunique cell ids')
    previous=Path.cwd();old_show=plt.show
    namespace={'__name__':'__overview__'}
    count=0;images=0
    try:
        os.chdir(root)
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            source=cell['source']
            if isinstance(source,list):source=''.join(source)
            count+=1;outputs=[];buffer=io.StringIO()
            def flush_text() -> None:
                text=buffer.getvalue()
                if text:
                    outputs.append(dict(output_type='stream',name='stdout',text=text))
                    buffer.seek(0);buffer.truncate(0)
            def show(*args,**kwargs) -> None:
                nonlocal images
                flush_text()
                for number in plt.get_fignums():
                    figure=plt.figure(number);data=io.BytesIO()
                    figure.savefig(data,format='png',dpi=110,bbox_inches='tight')
                    outputs.append(dict(output_type='display_data',metadata={},data={
                        'image/png':base64.b64encode(data.getvalue()).decode('ascii'),
                        'text/plain':'Matplotlib figure from executed overview cell'}))
                    images+=1;plt.close(figure)
            plt.show=show
            with redirect_stdout(buffer):exec(compile(source,f'{path.name}:cell-{count}','exec'),namespace)
            flush_text();cell['outputs']=outputs;cell['execution_count']=count
        notebook['metadata']['language_info']['version']=platform.python_version()
        notebook['metadata']['execution_note']='Code cells executed sequentially by experiments.check_presentation with ordinary Python; Jupyter kernel not executed.'
        path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
        snippets=re.findall(r'```python\n(.*?)\n```',(root/'README.md').read_text(),flags=re.S)
        if not snippets:raise ValueError('no README examples found')
        readme_namespace={'__name__':'__readme__'};readme_output=io.StringIO()
        with redirect_stdout(readme_output):
            for index,source in enumerate(snippets):
                exec(compile(source,f'README.md:example-{index+1}','exec'),readme_namespace)
    finally:
        plt.show=old_show;plt.close('all');os.chdir(previous)
    report=dict(successful=True,python=platform.python_version(),notebook_code_cells=count,
        embedded_figures=images,readme_python_blocks=len(snippets),readme_stdout=readme_output.getvalue(),
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),jupyter_kernel_executed=False,
        execution='Trusted repository cells executed in order with Python exec, explicit stdout and plt.show capture; no notebook-only magics.',
        numerical_scope='Single-seed examples and saved-table presentation; not a new statistical experiment.')
    destination=root/'results/raw/milestone_17_presentation_check.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('readme_stdout','notebook_sha256')},indent=2))


if __name__=='__main__':main()
