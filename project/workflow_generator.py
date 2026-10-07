import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


fig, ax = plt.subplots(1,1,figsize=(16,18))
ax.set_xlim(0,16); ax.set_ylim(0,18); ax.axis('off')
fig.patch.set_facecolor('white')

def box(ax,x,y,w,h,label,sub='',fc='#c8b8f0',ec='#6a3ab0',fs=12,sf=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=fc,edgecolor=ec,linewidth=1.5))
    ax.text(x+w/2,y+h*0.65,label,ha='center',va='center',fontsize=fs,fontweight='bold')
    if sub:
        [ax.text(x+w/2,y+h*0.32-i*0.18,l,ha='center',va='center',fontsize=sf,color='#333') for i,l in enumerate(sub.split('\n'))]

def arr(ax,x1,y1,x2,y2,c='#333',ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=c,lw=1.5,linestyle=ls))

def tbox(ax,x,y,title,tools,fc,ec):
    h=0.28*(len(tools)+1)+0.3
    ax.add_patch(FancyBboxPatch((x,y),4.2,h,boxstyle='round,pad=0.05',facecolor=fc,edgecolor=ec,linewidth=1))
    ax.text(x+2.1,y+h-0.22,title,ha='center',va='center',fontsize=9,fontweight='bold')
    for i,(t,helper) in enumerate(tools):
        ax.text(x+0.15,y+h-0.5-i*0.28,t,va='center',fontsize=8,fontweight='bold',color=ec)
        ax.text(x+0.15,y+h-0.7-i*0.28,helper,va='center',fontsize=7,color='#555')

ax.text(8,17.6,"Beaver's Choice Paper Company",ha='center',fontsize=16,fontweight='bold')
ax.text(8,17.3,'Multi-Agent System - smolagents ToolCallingAgent',ha='center',fontsize=11,color='#555')
box(ax,5.5,16.5,5,0.65,'Customer Request','request + (Date of request: YYYY-MM-DD)','#d0d0e8','#4a4a8a',12,8)
arr(ax,8,16.5,8,15.85)
box(ax,3,15.0,10,0.75,'Orchestrator Agent','Tools: run_inventory_check · run_quoting · run_sales_processing\nLLM decides sequence of tool calls','#c8b8f0','#6a3ab0',13,9)
arr(ax,5,15.0,2.5,13.65); ax.text(2.8,14.4,'1 inventory\n  report',fontsize=8)
arr(ax,8,15.0,8,13.65); ax.text(8.1,14.35,'2\nquote',fontsize=8)
arr(ax,11,15.0,13.5,13.65); ax.text(12.2,14.4,'3 order\n  confirm',fontsize=8)
box(ax,0.2,12.7,4.5,0.85,'Inventory Agent','run_inventory_check','#a8e6cf','#1a7a4a',12,9)
box(ax,5.8,12.7,4.5,0.85,'Quoting Agent','run_quoting','#ffd59e','#a06010',12,9)
box(ax,11.3,12.7,4.5,0.85,'Sales Agent','run_sales_processing','#ffb3b3','#a01010',12,9)
tbox(ax,0.2,9.8,'Inventory Tools',[('get_inventory_snapshot','-> get_all_inventory(as_of_date)'),('check_item_stock','-> get_stock_level(item_name, date)'),('estimate_delivery','-> get_supplier_delivery_date(date, qty)')],'#e8f8f0','#1a7a4a')
tbox(ax,5.8,9.8,'Quoting Tools',[('lookup_quote_history','-> search_quote_history(terms, limit)'),('get_company_financials','-> generate_financial_report(date)'),('calculate_bulk_discount','-> <500=0% <1k=5% <5k=10% >=5k=15%')],'#fff8ec','#a06010')
tbox(ax,11.3,9.8,'Sales Tools',[('check_cash_balance','-> get_cash_balance(as_of_date)'),('record_sale','-> create_transaction(item,type,qty,price,date)'),('get_delivery_estimate','-> get_supplier_delivery_date(date, qty)')],'#fff0f0','#a01010')
for x in [2.45,8.05,13.55]: arr(ax,x,12.7,x,11.0,c='#888',ls='dashed')
arr(ax,2.45,13.55,5.5,14.6,c='#1a7a4a',ls='dashed')
arr(ax,8.05,13.55,8.05,14.6,c='#a06010',ls='dashed')
arr(ax,13.55,13.55,10.5,14.6,c='#a01010',ls='dashed')
diamond=plt.Polygon([[8,9.4],[9.8,8.8],[8,8.2],[6.2,8.8]],closed=True,facecolor='#f5f5e8',edgecolor='#555',linewidth=1.5)
ax.add_patch(diamond)
ax.text(8,8.9,'Stock available?',ha='center',fontsize=11,fontweight='bold')
arr(ax,8,9.8,8,9.4)
arr(ax,9.8,8.8,11.2,8.8,c='#1a7a4a'); ax.text(10.2,9.0,'Yes',fontsize=11,color='#1a7a4a',fontweight='bold')
box(ax,11.2,8.45,2.5,0.7,'Order Confirmed','','#a8e6cf','#1a7a4a',11)
arr(ax,6.2,8.8,4.8,8.8,c='#a01010'); ax.text(5.1,9.0,'No',fontsize=11,color='#a01010',fontweight='bold')
box(ax,2.3,8.45,2.5,0.7,'Order Rejected','','#ffb3b3','#a01010',11)
arr(ax,12.45,8.45,9.5,7.55); arr(ax,3.55,8.45,6.5,7.55)
box(ax,5.5,6.8,5,0.65,'Customer Response','confirmation · pricing · delivery date · rejection reason','#d0d0e8','#4a4a8a',12,8)
box(ax,6.0,5.6,4,0.6,'SQLite Database','transactions · inventory · quotes','#e8e8f8','#4a4a8a',11,8)
arr(ax,8,6.8,8,6.2,c='#888',ls='dashed')
ax.add_patch(FancyBboxPatch((0.2,4.0),15.6,1.4,boxstyle='round,pad=0.1',facecolor='#f8f8f8',edgecolor='#ccc'))
ax.text(0.5,5.2,'Legend:',fontsize=10,fontweight='bold')
ax.annotate('',xy=(2.5,4.95),xytext=(1.5,4.95),arrowprops=dict(arrowstyle='->',color='#333',lw=1.5))
ax.text(2.6,4.95,'Task delegation (orchestrator to agent)',va='center',fontsize=9)
ax.annotate('',xy=(2.5,4.6),xytext=(1.5,4.6),arrowprops=dict(arrowstyle='->',color='#888',lw=1.2,linestyle='dashed'))
ax.text(2.6,4.6,'Result returned / tool call to helper function',va='center',fontsize=9)
ax.text(6.5,4.95,'Each @tool wraps exactly one helper function from starter code',va='center',fontsize=9)
ax.text(6.5,4.6,'orchestrator_agent delegates via run_inventory_check, run_quoting, run_sales_processing',va='center',fontsize=8,color='#555')
plt.tight_layout()
plt.savefig('agent_workflow.png',dpi=150,bbox_inches='tight',facecolor='white')
print('Done! agent_workflow.png saved in project folder')
