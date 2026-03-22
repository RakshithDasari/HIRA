import streamlit as st
import os, sys, json, time, pickle, tempfile

st.set_page_config(page_title="HIRA", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

def get_theme(dark):
    if dark:
        return """
:root{
    --bg:#1a1b1e;--bg2:#202228;--bg3:#272a31;--bg4:#2f333c;
    --border:#2e3138;--border2:#383d47;
    --text:#e8eaed;--text2:#a8acb4;--text3:#6b7080;--text4:#474c58;
    --gold:#d4a853;--goldd:rgba(212,168,83,0.13);
    --blue:#6b9fd4;--blued:rgba(107,159,212,0.12);
    --green:#5fa876;--greend:rgba(95,168,118,0.12);
    --red:#c86b6b;--redd:rgba(200,107,107,0.12);
}"""
    else:
        return """
:root{
    --bg:#f8f9fa;--bg2:#ffffff;--bg3:#e8ebef;--bg4:#dde1e7;
    --border:#e0e4ea;--border2:#cdd2da;
    --text:#1a1c20;--text2:#404550;--text3:#6b7280;--text4:#9aa0aa;
    --gold:#9a6820;--goldd:rgba(154,104,32,0.1);
    --blue:#2e6a96;--blued:rgba(46,106,150,0.1);
    --green:#2e7a48;--greend:rgba(46,122,72,0.1);
    --red:#963030;--redd:rgba(150,48,48,0.1);
}"""

BASE = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@300;400;450;500&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root {
    /* Persistent dark theme variables for the sidebar */
    --sb-bg:#1a1b1e; --sb-bg2:#202228; --sb-bg3:#272a31; --sb-bg4:#2f333c;
    --sb-border:#2e3138; --sb-border2:#383d47;
    --sb-text:#e8eaed; --sb-text2:#a8acb4; --sb-text3:#6b7080; --sb-text4:#474c58;
}

html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;font-size:14px;background:var(--bg);color:var(--text);-webkit-font-smoothing:antialiased;}
/* Unhide header but make it transparent to keep Streamlit's native sidebar toggle button */
#MainMenu,footer{display:none!important} header{background:transparent!important;}
.block-container{padding:0!important;max-width:100%!important}
.stApp{background:var(--bg)}

/* SIDEBAR - ALWAYS DARK */
section[data-testid="stSidebar"]{background:var(--sb-bg)!important;border-right:1px solid var(--sb-border)!important;min-width:260px!important;max-width:260px!important;}
section[data-testid="stSidebar"]>div{padding:0!important}

.logo-wrap{padding:28px 22px 22px;border-bottom:1px solid var(--sb-border)}
.logo-eyebrow{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:500;letter-spacing:0.22em;color:var(--gold);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:7px;}
.logo-eyebrow::before{content:'';width:14px;height:1px;background:var(--gold);display:inline-block}
.logo-name{font-family:'Lora',Georgia,serif;font-size:44px;font-weight:600;letter-spacing:-0.02em;line-height:1;color:var(--sb-text);margin-bottom:12px;}
.logo-name span{color:var(--gold)}
.logo-desc{font-size:11px;color:var(--sb-text3);line-height:1.7;font-style:italic;font-family:'Lora',serif;}
.nav-wrap{padding:14px 10px 8px}
.nav-label{font-size:9px;font-weight:500;letter-spacing:0.16em;text-transform:uppercase;color:var(--sb-text4);padding:4px 12px 8px;}
.kb-wrap{padding:14px 22px}
.kb-status{font-size:11.5px;color:var(--sb-text3);display:flex;align-items:center;gap:7px;margin-bottom:11px;}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot-g{background:var(--green)}.dot-a{background:var(--gold)}
.kb-row{display:flex;gap:8px}
.kb-card{flex:1;background:var(--sb-bg2);border:1px solid var(--sb-border);border-radius:8px;padding:10px 12px;text-align:center;}
.kb-n{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:500;color:var(--gold);line-height:1.1;}
.kb-l{font-size:9px;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:var(--sb-text4);margin-top:3px}
.sb-foot{padding:14px 22px;border-top:1px solid var(--sb-border)}
.sb-stack{font-size:10.5px;color:var(--sb-text4);line-height:1.8}

/* Sidebar Specific Buttons - WEB3 STYLE */
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(145deg, var(--sb-bg2) 0%, var(--sb-bg3) 100%) !important;
    color: var(--sb-text) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(145deg, var(--sb-bg3) 0%, var(--sb-bg4) 100%) !important;
    border: 1px solid rgba(212, 168, 83, 0.4) !important;
    box-shadow: 0 0 15px rgba(212, 168, 83, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-2px) !important;
    color: var(--gold) !important;
}
section[data-testid="stSidebar"] .stButton>button:active {
    transform: translateY(1px) !important;
}

/* HERO — normal flow centering */
.hero-shell{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    margin-top:22vh;
    margin-bottom:32px;
}
.hero-inner{
    width:100%;
    max-width:680px;
    text-align:center;
}
.hero-title{
    font-family:'Lora',Georgia,serif;
    font-size:56px;
    font-weight:500;
    letter-spacing:-0.02em;
    line-height:1.05;
    color:var(--text);
    margin-bottom:12px;
}
.hero-title span{color:var(--gold)}
.hero-sub{
    font-size:12px;
    font-weight:500;
    color:var(--text4);
    letter-spacing:0.14em;
    text-transform:uppercase;
}

/* CHAT layout */
.main-wrap{padding:52px 64px 100px;max-width:900px;margin:0 auto}
.ph{margin-bottom:32px}
.ph-title{font-family:'Lora',Georgia,serif;font-size:40px;font-weight:500;color:var(--text);letter-spacing:-0.02em;line-height:1.1;margin-bottom:8px;}
.ph-title span{color:var(--gold)}
.ph-sub{font-size:11px;font-weight:500;color:var(--text4);letter-spacing:0.12em;text-transform:uppercase;}

.msg-u{display:flex;justify-content:flex-end;padding:16px 0 4px}
.msg-ubub{background:var(--bg3);border:1px solid var(--border2);border-radius:18px 18px 4px 18px;padding:12px 16px;max-width:66%;font-size:14px;line-height:1.7;color:var(--text);}
.msg-h{padding:20px 0 24px;border-bottom:1px solid var(--border)}
.msg-hl{font-size:9.5px;font-weight:500;letter-spacing:0.18em;text-transform:uppercase;color:var(--gold);display:flex;align-items:center;gap:8px;margin-bottom:14px;}
.msg-hl::before{content:'◈';font-size:10px}
.tc{background:var(--bg2);border:1px solid var(--border);border-left:2px solid var(--blue);border-radius:0 8px 8px 0;padding:10px 14px;margin:4px 0;font-family:'JetBrains Mono',monospace;font-size:11px;}
.tc-t{color:var(--blue);font-size:9px;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:4px}
.tc-q{color:var(--text2);margin:3px 0}.tc-f{color:var(--text4);font-size:10.5px;margin-top:2px}
.ans{font-size:14.5px;line-height:1.82;color:var(--text)}

/* CARDS */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px 22px;margin-bottom:12px;}
.card-t{font-size:9.5px;font-weight:500;letter-spacing:0.15em;text-transform:uppercase;color:var(--text4);margin-bottom:14px;display:flex;align-items:center;gap:6px;}
.card-t::before{content:'';display:inline-block;width:12px;height:1px;background:var(--text4)}
.card-row{font-size:13px;color:var(--text2);line-height:2.0}
.card-gold{background:var(--goldd);border:1px solid rgba(212,168,83,0.3);border-radius:10px;padding:18px 20px;margin-bottom:12px;}
.step{display:flex;align-items:center;gap:10px;padding:9px 0;font-size:13px;color:var(--text4);border-bottom:1px solid var(--border);}
.step:last-child{border-bottom:none}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--border2);flex-shrink:0;transition:background 0.3s}
.step.done{color:var(--text2)}.step.done .sdot{background:var(--green)}
.step.act{color:var(--gold)}.step.act .sdot{background:var(--gold)}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:10.5px;font-weight:450;letter-spacing:0.02em;}
.bg{background:var(--greend);color:var(--green);border:1px solid var(--green)}
.bgold{background:var(--goldd);color:var(--gold);border:1px solid var(--gold)}
.bb{background:var(--blued);color:var(--blue);border:1px solid var(--blue)}
.br{background:var(--redd);color:var(--red);border:1px solid var(--red)}
.lim{font-size:11.5px;color:var(--text4);padding:7px 11px;background:var(--bg2);border:1px solid var(--border);border-radius:6px;margin-top:8px;}
.div{height:1px;background:var(--border);margin:24px 0}
.empty-state{padding:72px 40px;text-align:center;border:1px dashed var(--border2);border-radius:14px;}
.empty-icon{font-size:28px;margin-bottom:16px;opacity:0.35;color:var(--gold);font-family:'JetBrains Mono',monospace}
.empty-title{font-family:'Lora',serif;font-size:20px;color:var(--text2);margin-bottom:8px}
.empty-sub{font-size:13px;color:var(--text3);line-height:1.65}
.graph-outer{border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-top:4px;background:var(--bg2);}
.graph-head{padding:12px 18px;border-bottom:1px solid var(--border);font-size:11.5px;font-weight:500;color:var(--text3);display:flex;justify-content:space-between;align-items:center;}
iframe{border:none!important;display:block!important}

/* STREAMLIT (Main content specific) - WEB3 BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, var(--bg3) 0%, var(--bg2) 100%) !important;
    color: var(--text) !important;
    border: 1px solid rgba(107, 159, 212, 0.3) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, var(--bg4) 0%, var(--bg3) 100%) !important;
    border: 1px solid var(--blue) !important;
    box-shadow: 0 0 20px var(--blued), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-2px) !important;
    color: var(--blue) !important;
}
.stButton>button:active {
    transform: translateY(1px) !important;
}

.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;font-family:'Inter',sans-serif!important;font-size:15px!important;padding:14px 18px!important;box-shadow:none!important;}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px var(--goldd)!important;}
.stSelectbox>div>div{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:9px!important;color:var(--text)!important;font-size:13px!important;}
.stFileUploader{background:var(--bg2)!important;border:1.5px dashed var(--border2)!important;border-radius:10px!important;}
.stFileUploader label,.stFileUploader section p{color:var(--text3)!important;font-size:13px!important}
.stProgress>div>div>div{background:var(--gold)!important}
.stProgress>div>div{background:var(--bg3)!important}
div[data-testid="stExpander"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:9px!important}
div[data-testid="stExpander"] summary{color:var(--text2)!important;font-size:13px!important}
.stSuccess{background:var(--greend)!important;color:var(--green)!important;border:1px solid var(--green)!important;border-radius:8px!important}
.stError{background:var(--redd)!important;color:var(--red)!important;border:1px solid var(--red)!important;border-radius:8px!important}
.stWarning{background:var(--goldd)!important;color:var(--gold)!important;border:1px solid var(--gold)!important;border-radius:8px!important}
.stInfo{background:var(--blued)!important;color:var(--blue)!important;border:1px solid var(--blue)!important;border-radius:8px!important}
hr{border-color:var(--border)!important;margin:16px 0!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}
"""

for k,v in {"dark":True,"mode":"query","messages":[],"artifacts_exist":os.path.exists("artifacts/metadata.json")}.items():
    if k not in st.session_state: st.session_state[k]=v

st.markdown(f"<style>{get_theme(st.session_state.dark)}{BASE}</style>", unsafe_allow_html=True)

def load_meta():
    try:
        with open("artifacts/metadata.json") as f: return json.load(f)
    except: return {}
def size_ok(sz,ext): return sz<={".txt":5<<20,".pdf":20<<20}.get(ext,5<<20)
def size_lbl(ext): return {".txt":"5 MB",".pdf":"20 MB"}.get(ext,"5 MB")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
        <div class="logo-eyebrow">System</div>
        <div class="logo-name">H<span>I</span>RA</div>
        <div class="logo-desc">Hypergraph-Indexed Retrieval Augmentation via Multimodal Agentic Reasoning</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="nav-wrap"><div class="nav-label">Navigation</div></div>', unsafe_allow_html=True)
    for key,icon,label in [("query","◎","Query"),("build","⊕","Build"),("update","↺","Update"),("visualize","◫","Visualize")]:
        if st.button(f"{icon}  {label}",key=f"n_{key}",use_container_width=True):
            st.session_state.mode=key; st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    meta=load_meta()
    if meta:
        e=meta.get("entity_count",0); h=meta.get("hyperedge_count",0)
        st.markdown(f"""<div class="kb-wrap"><div class="kb-status"><span class="dot dot-g"></span>Knowledge base active</div><div class="kb-row"><div class="kb-card"><div class="kb-n">{e}</div><div class="kb-l">Entities</div></div><div class="kb-card"><div class="kb-n">{h}</div><div class="kb-l">Facts</div></div></div></div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="kb-wrap"><div class="kb-status"><span class="dot dot-a"></span>No knowledge base yet</div></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("☀  Light" if st.session_state.dark else "☽  Dark",key="theme",use_container_width=True):
        st.session_state.dark=not st.session_state.dark; st.rerun()
    st.markdown('<div class="sb-foot"><div class="sb-stack">Gemini Embedding 2 · Nvidia Nemotron<br>NetworkX · FAISS · OpenRouter</div></div>', unsafe_allow_html=True)

# ── QUERY ─────────────────────────────────────────────────────────────────────
if st.session_state.mode=="query":
    if not st.session_state.messages:
        # Normal document flow centering
        st.markdown("""
        <div class="hero-shell">
            <div class="hero-inner">
                <div class="hero-title">Ask <span>HIRA</span></div>
                <div class="hero-sub">Multimodal agentic knowledge retrieval</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Centered input columns
        _, mid, _ = st.columns([1, 5, 1])
        with mid:
            q = st.text_input("", "", placeholder="Ask anything about your knowledge base...", label_visibility="collapsed", key="qi")
            c1, c2 = st.columns([5, 1])
            with c2:
                send = st.button("Send →", key="send", use_container_width=True)
            with c1:
                st.markdown('<div style="font-size:11px;color:var(--text4);padding-top:11px;text-align:center">No knowledge base yet? Use ⊕ Build to get started.</div>', unsafe_allow_html=True)

        if send and q:
            if not st.session_state.artifacts_exist:
                st.error("No knowledge base found. Build one first using ⊕ Build.")
            else:
                st.session_state.messages.append({"role":"user","content":q})
                with st.spinner("Reasoning over knowledge graph..."):
                    try:
                        sys.path.insert(0,os.getcwd())
                        from agent.retriever import Retriever
                        from agent.controller import Controller
                        retriever=Retriever(); controller=Controller(max_turns=3)
                        turns_log=[]; cq=q
                        for turn in range(3):
                            ctx=retriever.search(cq)
                            turns_log.append({"turn":turn+1,"query":cq,"entities":ctx.get("entity_count",0),"facts":ctx.get("fact_count",0)})
                            dec=controller.decide(q,ctx)
                            if dec["done"]: answer=dec["answer"]; break
                            cq=dec.get("next_query",q)
                        else:
                            answer=controller.decide(q,ctx).get("answer") or "Could not find a sufficient answer."
                        st.session_state.messages.append({"role":"assistant","content":answer,"turns":turns_log})
                    except Exception as e:
                        st.session_state.messages.append({"role":"assistant","content":f"Error: {e}","turns":[]})
                st.rerun()
    else:
        st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="ph"><div class="ph-title">Ask <span>HIRA</span></div><div class="ph-sub">Multimodal agentic knowledge retrieval</div></div>', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"]=="user":
                st.markdown(f'<div class="msg-u"><div class="msg-ubub">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                turns="".join([f'<div class="tc"><div class="tc-t">Turn {t["turn"]} — searching</div><div class="tc-q">"{t["query"]}"</div><div class="tc-f">{t["entities"]} entities · {t["facts"]} facts</div></div>' for t in msg.get("turns",[])])
                st.markdown(f'<div class="msg-h"><div class="msg-hl">HIRA</div>{turns}<div class="ans">{msg["content"]}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        c1,c2=st.columns([8,1],gap="small")
        with c1:
            q=st.text_input("","",placeholder="Ask a follow-up...",label_visibility="collapsed",key="qi2")
        with c2:
            send=st.button("Send",key="send2",use_container_width=True)
        if send and q:
            st.session_state.messages.append({"role":"user","content":q})
            with st.spinner("Reasoning..."):
                try:
                    sys.path.insert(0,os.getcwd())
                    from agent.retriever import Retriever
                    from agent.controller import Controller
                    retriever=Retriever(); controller=Controller(max_turns=3)
                    turns_log=[]; cq=q
                    for turn in range(3):
                        ctx=retriever.search(cq)
                        turns_log.append({"turn":turn+1,"query":cq,"entities":ctx.get("entity_count",0),"facts":ctx.get("fact_count",0)})
                        dec=controller.decide(q,ctx)
                        if dec["done"]: answer=dec["answer"]; break
                        cq=dec.get("next_query",q)
                    else:
                        answer=controller.decide(q,ctx).get("answer") or "Could not find a sufficient answer."
                    st.session_state.messages.append({"role":"assistant","content":answer,"turns":turns_log})
                except Exception as e:
                    st.session_state.messages.append({"role":"assistant","content":f"Error: {e}","turns":[]})
            st.rerun()
        if st.button("Clear conversation",key="clr"): st.session_state.messages=[]; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ── BUILD ─────────────────────────────────────────────────────────────────────
elif st.session_state.mode=="build":
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="ph"><div class="ph-title">Build</div><div class="ph-sub">Index a document into the knowledge base</div></div>', unsafe_allow_html=True)
    col1,col2=st.columns([3,2],gap="large")
    with col1:
        up=st.file_uploader("",type=["txt","pdf"],label_visibility="collapsed")
        st.markdown('<div class="lim">Supported: .txt (max 5 MB) · .pdf (max 20 MB, images extracted automatically)</div>', unsafe_allow_html=True)
        st.selectbox("Embedding model",["Gemini Embedding 2 — multimodal, 3072-dim","Nvidia Nemotron Embed VL 1B — document-specialized, free"])
        if up:
            ext=os.path.splitext(up.name)[1].lower()
            if not size_ok(up.size,ext): st.error(f"File exceeds {size_lbl(ext)} limit.")
            else:
                st.markdown(f'<div class="badge bg" style="margin:12px 0 18px">✓  {up.name} &nbsp;·&nbsp; {up.size/1024:.1f} KB</div>', unsafe_allow_html=True)
                if st.button("⊕  Build knowledge base",use_container_width=True,key="bb"):
                    sp=f"data/sample/{up.name}"; os.makedirs("data/sample",exist_ok=True)
                    with open(sp,"wb") as f: f.write(up.getbuffer())
                    steps=["Reading and chunking document","Extracting entities and hyperedges","Encoding with embedding model","Building NetworkX hypergraph","Indexing with FAISS","Saving artifacts to disk"]
                    prog=st.progress(0); ph=st.empty()
                    def rs(ai):
                        h="".join([f'<div class="step {"done" if j<ai else "act" if j==ai else ""}"><div class="sdot"></div>{"✓ " if j<ai else "→ " if j==ai else ""}{s}</div>' for j,s in enumerate(steps)])
                        ph.markdown(h,unsafe_allow_html=True)
                    try:
                        sys.path.insert(0,os.getcwd())
                        from graph.builder import build
                        for i in range(len(steps)): rs(i); prog.progress((i+1)/len(steps)); time.sleep(0.25)
                        build(sp); rs(len(steps)); prog.progress(1.0)
                        st.session_state.artifacts_exist=True; st.success("Knowledge base built."); st.rerun()
                    except Exception as e: st.error(f"Build failed: {e}")
    with col2:
        st.markdown("""
        <div class="card"><div class="card-t">Pipeline steps</div><div class="card-row">① Chunk (500 words, 100 overlap)<br>② Nvidia Nemotron extracts entities<br>③ Embedding model encodes nodes<br>④ NetworkX builds hypergraph<br>⑤ FAISS indexes vectors<br>⑥ 4 artifacts saved to disk</div></div>
        <div class="card-gold"><div class="card-t" style="color:var(--gold)">Embedding model guide</div><div class="card-row"><b>Gemini Embedding 2</b><br>Best for text + image corpora<br>3072-dim unified vector space<br><br><b>Nvidia Nemotron VL 1B</b><br>Best for document pipelines<br>Completely free, no API cost</div></div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── UPDATE ────────────────────────────────────────────────────────────────────
elif st.session_state.mode=="update":
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="ph"><div class="ph-title">Update</div><div class="ph-sub">Incremental knowledge addition — never rebuilds</div></div>', unsafe_allow_html=True)
    if not st.session_state.artifacts_exist:
        st.markdown('<div class="empty-state"><div class="empty-icon">↺</div><div class="empty-title">No knowledge base found</div><div class="empty-sub">Build a knowledge base first.</div></div>', unsafe_allow_html=True)
    else:
        meta=load_meta(); col1,col2=st.columns([3,2],gap="large")
        with col1:
            up=st.file_uploader("",type=["txt","pdf"],label_visibility="collapsed",key="uu")
            st.markdown('<div class="lim">Supported: .txt (max 5 MB) · .pdf (max 20 MB)</div>', unsafe_allow_html=True)
            if up:
                ext=os.path.splitext(up.name)[1].lower()
                if not size_ok(up.size,ext): st.error(f"File exceeds {size_lbl(ext)} limit.")
                else:
                    st.markdown(f'<div class="badge bg" style="margin:12px 0 18px">✓  {up.name} &nbsp;·&nbsp; {up.size/1024:.1f} KB</div>', unsafe_allow_html=True)
                    if st.button("↺  Update knowledge base",use_container_width=True,key="ub"):
                        sp=f"data/sample/{up.name}"; os.makedirs("data/sample",exist_ok=True)
                        with open(sp,"wb") as f: f.write(up.getbuffer())
                        with st.spinner("Computing diff..."):
                            try:
                                sys.path.insert(0,os.getcwd())
                                from graph.updater import Updater
                                r=Updater().update(sp); ae,ah=r["added_entities"],r["added_hyperedges"]
                                if ae==0 and ah==0: st.info("No new knowledge found.")
                                else: st.success(f"+{ae} entities · +{ah} hyperedges added")
                                st.session_state.artifacts_exist=True; st.rerun()
                            except Exception as e: st.error(f"Update failed: {e}")
        with col2:
            e=meta.get("entity_count",0); h=meta.get("hyperedge_count",0)
            st.markdown(f"""
            <div class="card"><div class="card-t">Current knowledge base</div><div style="display:flex;gap:16px;margin-bottom:4px"><div><div class="kb-n">{e}</div><div class="kb-l">Entities</div></div><div><div class="kb-n" style="color:var(--blue)">{h}</div><div class="kb-l">Hyperedges</div></div></div></div>
            <div class="card"><div class="card-t">How diff works</div><div class="card-row">Compare by entity name (O(1) set)<br>Skip all duplicates automatically<br>Append new vectors to FAISS only<br>Cost ∝ new data, not total corpus</div></div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── VISUALIZE ─────────────────────────────────────────────────────────────────
elif st.session_state.mode=="visualize":
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="ph"><div class="ph-title">Visualize</div><div class="ph-sub">Explore the knowledge hypergraph</div></div>', unsafe_allow_html=True)
    if not st.session_state.artifacts_exist:
        st.markdown('<div class="empty-state"><div class="empty-icon">◫</div><div class="empty-title">No knowledge base found</div><div class="empty-sub">Build a knowledge base first.</div></div>', unsafe_allow_html=True)
    else:
        try:
            import networkx as nx
            with open("artifacts/graph.gpickle","rb") as f: G=pickle.load(f)
            meta=load_meta(); entities=meta.get("entities",[]); hyperedges=meta.get("hyperedges",[])
            st.markdown(f"""<div style="display:flex;gap:10px;margin-bottom:32px;flex-wrap:wrap"><span class="badge bb">{G.number_of_nodes()} nodes</span><span class="badge bgold">{G.number_of_edges()} edges</span><span class="badge bg">{len(entities)} entities</span><span class="badge bb">{len(hyperedges)} hyperedges</span></div>""", unsafe_allow_html=True)
            c1,c2=st.columns(2,gap="large")
            with c1:
                with st.expander(f"Entities ({len(entities)})",expanded=True):
                    for e in entities[:50]:
                        st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center"><span style="font-size:13px;color:var(--text)">{e["name"]}</span><span class="badge bgold" style="font-size:10px">{e.get("type","concept")}</span></div>', unsafe_allow_html=True)
                    if len(entities)>50: st.markdown(f'<div style="font-size:11px;color:var(--text4);padding-top:8px">+ {len(entities)-50} more</div>', unsafe_allow_html=True)
            with c2:
                with st.expander(f"Hyperedges ({len(hyperedges)})",expanded=True):
                    for h in hyperedges[:20]:
                        connects=", ".join(h.get("connects",[])[:3])
                        st.markdown(f'<div style="padding:8px 0;border-bottom:1px solid var(--border)"><div style="font-size:13px;color:var(--text);margin-bottom:3px">{h["fact"][:80]}{"..." if len(h["fact"])>80 else ""}</div><div style="font-size:10.5px;color:var(--text4)">{connects}</div></div>', unsafe_allow_html=True)
                    if len(hyperedges)>20: st.markdown(f'<div style="font-size:11px;color:var(--text4);padding-top:8px">+ {len(hyperedges)-20} more</div>', unsafe_allow_html=True)
            st.markdown('<div class="div"></div><div style="font-size:15px;font-weight:500;color:var(--text2);margin-bottom:16px">Network graph</div>', unsafe_allow_html=True)
            try:
                from pyvis.network import Network
                import streamlit.components.v1 as components
                nodes_list=list(G.nodes())[:80]; subG=G.subgraph(nodes_list)
                bg="#1a1b1e" if st.session_state.dark else "#ffffff"
                fc="#e8eaed" if st.session_state.dark else "#1a1c20"
                net=Network(height="650px",width="100%",bgcolor=bg,font_color=fc)
                net.from_nx(subG)
                for node in net.nodes:
                    nt=G.nodes[node["id"]].get("type","entity")
                    if nt=="entity": node["color"]="#d4a853"; node["size"]=28; node["shape"]="dot"; node["font"]={"size":14,"color":fc}
                    else:
                        node["color"]="#6b9fd4"; node["size"]=18; node["shape"]="square"
                        lbl=str(node["id"]); node["label"]=lbl[:28]+"..." if len(lbl)>28 else lbl; node["font"]={"size":12,"color":fc}
                edge_color='#3a3e48' if st.session_state.dark else '#cdd2da'
                net.set_options(f'{{"physics":{{"barnesHut":{{"gravitationalConstant":-8000,"springLength":160,"springConstant":0.04}},"stabilization":{{"iterations":120}}}},"edges":{{"color":{{"color":"{edge_color}"}},"width":1.5,"smooth":{{"type":"continuous"}}}},"interaction":{{"hover":true,"zoomView":true,"dragView":true,"navigationButtons":true,"keyboard":true}},"nodes":{{"borderWidth":0}}}}')
                with tempfile.NamedTemporaryFile(mode='w',suffix='.html',delete=False,encoding='utf-8') as tf: tmp_path=tf.name
                net.save_graph(tmp_path)
                with open(tmp_path,encoding='utf-8') as f: hc=f.read()
                os.unlink(tmp_path)
                hc=hc.replace('<body>',f'<body style="background:{bg};margin:0;padding:0">')
                hc=hc.replace('<html>',f'<html style="background:{bg}">')
                st.markdown('<div class="graph-outer"><div class="graph-head"><span>Knowledge hypergraph — scroll to zoom · drag to pan</span><span style="display:flex;gap:8px"><span class="badge bgold">● entity</span><span class="badge bb">■ hyperedge</span></span></div>', unsafe_allow_html=True)
                components.html(hc,height=660)
                st.markdown('</div>', unsafe_allow_html=True)
                if G.number_of_nodes()>80: st.markdown(f'<div class="lim">Showing 80 of {G.number_of_nodes()} nodes.</div>', unsafe_allow_html=True)
            except ImportError:
                st.markdown('<div class="lim">Install pyvis: <code>pip install pyvis</code></div>', unsafe_allow_html=True)
        except Exception as e: st.error(f"Failed to load graph: {e}")
    st.markdown('</div>', unsafe_allow_html=True)