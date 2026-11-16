"""Execute tutorials in-process without requiring localhost kernel sockets.
Uses IPython's execution/display machinery and records actual cell outputs.
"""
from pathlib import Path
import os,base64,io
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    book=nbformat.read(path,as_version=4)
    shell=InteractiveShell()
    count=0
    for cell in book.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            outcome=shell.run_cell(cell.source,store_history=False)
        if outcome.error_before_exec or outcome.error_in_exec:
            raise RuntimeError(f'{path.name}, cell {count}: {outcome.error_before_exec or outcome.error_in_exec}')
        cell.execution_count=count;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        for number in plt.get_fignums():
            buffer=io.BytesIO();plt.figure(number).savefig(buffer,format='png',bbox_inches='tight')
            cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()},metadata={}))
        plt.close('all')
    nbformat.write(book,path)
    print(f'Executed {path.name}: {count} code cells (in-process IPython)',flush=True)
