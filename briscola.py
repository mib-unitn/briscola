import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import ast
import altair as alt

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Briscola League",
    layout="wide",
    page_icon="♠️",
    initial_sidebar_state="expanded"
)

# --- 1. CONFIGURATION ---

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ea2DT-FlYmq_TjA4MFqdc1rr3GFj1MxpVfWm0MQxrQo/edit?gid=0#gid=0"

# NOTA: La LISTA_GIOCATORI ora viene caricata dinamicamente dal foglio "Giocatori".
# Se il foglio è vuoto, userà questa lista di default per inizializzarlo.
DEFAULT_GIOCATORI = ["Michele", "Federico", "Lorenza", "Grazia", "Pierpaolo", "Niccolò", "Simone", "Avinash", "Sahil", "Massine", "Beppe", "Esteban", "Giampaolo"]

PUNTI_MAP = {2: 2, 3: 3, 4: 2}
PUNTI_BONUS_100 = 0.5

ELO_STARTING = 1000
ELO_K_FACTOR = 32
PODIO_MIN_PG = 20
STATS_MIN_TOTAL_PG = 25
STATS_MIN_PAIR_PG = 10

# --- CONFIGURAZIONE DECADIMENTO (DECAY) ---
DECAY_GIORNI_GRAZIA = 21
DECAY_PUNTI_GIORNO = 1
DECAY_MIN_ELO = 800

COLONNE_LOG = ["data", "giorno_settimana", "giocatori", "vincitori", "num_giocatori", "punti_vittoria", "punti_bonus"]
COLONNE_CLASSIFICA = ["Giocatore", "PG", "V2", "V3", "V4", "PT", "MPP", "Elo", "UltimaPartita"]

# --- 2. CUSTOM CSS & STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .podium-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        width: 100%;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        transition: transform 0.3s ease;
    }
    
    .podium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 25px rgba(0,0,0,0.1);
    }

    .medal { font-size: 3rem; margin-bottom: 10px; }
    .player-name { font-size: 1.5rem; font-weight: 700; color: #333; margin: 5px 0; }
    .player-elo { font-size: 1.2rem; font-weight: 600; color: #FF4B4B; }
    .player-stats { font-size: 0.9rem; color: #888; }

    .gold { border-top: 6px solid #FFD700; background: linear-gradient(180deg, #fff 0%, #fffdf0 100%); }
    .silver { border-top: 6px solid #C0C0C0; background: linear-gradient(180deg, #fff 0%, #f8f9fa 100%); }
    .bronze { border-top: 6px solid #CD7F32; background: linear-gradient(180deg, #fff 0%, #fff5f0 100%); }
    
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND FUNCTIONS ---

def connect_to_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open_by_url(GOOGLE_SHEET_URL)
        return sh
    except Exception as e:
        st.error(f"GSheets Connection Error: {e}")
        return None

def get_worksheets(sh):
    """Restituisce i worksheet 'Log' e 'Giocatori', creandoli se non esistono."""
    # 1. LOG SHEET
    try:
        ws_log = sh.sheet1 
    except:
        ws_log = sh.get_worksheet(0)

    # 2. PLAYERS SHEET
    try:
        ws_players = sh.worksheet("Giocatori")
    except gspread.WorksheetNotFound:
        ws_players = sh.add_worksheet(title="Giocatori", rows=100, cols=1)
        # Inizializza con i default se creato nuovo
        df_init = pd.DataFrame(DEFAULT_GIOCATORI, columns=["Nome"])
        set_with_dataframe(ws_players, df_init, include_index=False, resize=True)
    
    return ws_log, ws_players

def carica_dati(ws_log, ws_players):
    # Carica Log
    try:
        df_log = get_as_dataframe(ws_log, evaluate_formulas=True, dtype_backend='pyarrow')
        if df_log.empty: df_log = pd.DataFrame(columns=COLONNE_LOG)
        df_log = df_log.dropna(how='all').dropna(axis=1, how='all')
        
        for col in ['giocatori', 'vincitori']:
            if col in df_log.columns:
                df_log[col] = df_log[col].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        
        for c in ['num_giocatori', 'punti_vittoria', 'punti_bonus']:
            if c in df_log.columns: df_log[c] = pd.to_numeric(df_log[c], errors='coerce').fillna(0)
            
        if 'data' in df_log.columns:
            df_log['data'] = pd.to_datetime(df_log['data'], errors='coerce')
        else:
            df_log['data'] = pd.NaT

        for col in COLONNE_LOG:
            if col not in df_log.columns: df_log[col] = pd.NA
            
        df_log = df_log[COLONNE_LOG].astype({'punti_bonus': 'float', 'punti_vittoria': 'float', 'num_giocatori': 'int'})
    except Exception:
        df_log = pd.DataFrame(columns=COLONNE_LOG)

    # Carica Giocatori
    try:
        df_pl = get_as_dataframe(ws_players, evaluate_formulas=True)
        if df_pl.empty or 'Nome' not in df_pl.columns:
            lista_giocatori = DEFAULT_GIOCATORI
        else:
            lista_giocatori = df_pl['Nome'].dropna().unique().tolist()
    except:
        lista_giocatori = DEFAULT_GIOCATORI

    return df_log, lista_giocatori

def salva_log_gsheet(df):
    if 'ws_log' in st.session_state and st.session_state.ws_log:
        try:
            df_save = df.copy()
            df_save['giocatori'] = df_save['giocatori'].astype(str)
            df_save['vincitori'] = df_save['vincitori'].astype(str)
            st.session_state.ws_log.clear()
            set_with_dataframe(st.session_state.ws_log, df_save, include_index=False, resize=True)
            return True
        except: return False
    return False

def salva_giocatori_gsheet(lista):
    if 'ws_players' in st.session_state and st.session_state.ws_players:
        try:
            df = pd.DataFrame(lista, columns=["Nome"])
            st.session_state.ws_players.clear()
            set_with_dataframe(st.session_state.ws_players, df, include_index=False, resize=True)
            return True
        except: return False
    return False

def calcola_stats_dettagliate(player, log):
    df_p = log[log['giocatori'].apply(lambda x: isinstance(x, list) and player in x)]
    if df_p.empty: return None

    totale = len(df_p)
    vittorie = len(df_p[df_p['vincitori'].apply(lambda x: isinstance(x, list) and player in x)])
    wr = (vittorie / totale) * 100

    if totale < STATS_MIN_TOTAL_PG:
        msg = "Not enough data"
        return {"wr": wr, "best_partner": msg, "nemesis": msg, "totale": totale}

    compagni, avversari = {}, {}

    for row in df_p.itertuples():
        # Safety checks
        if not isinstance(row.giocatori, list) or not isinstance(row.vincitori, list): continue
        
        win = player in row.vincitori
        if row.num_giocatori == 4:
            team = row.vincitori if win else [p for p in row.giocatori if p not in row.vincitori]
            partners = [p for p in team if p != player]
            if partners:
                p_name = partners[0]
                compagni.setdefault(p_name, [0, 0])
                compagni[p_name][1] += 1
                if win: compagni[p_name][0] += 1

        opps = [p for p in row.giocatori if p not in row.vincitori] if win else row.vincitori
        for o in opps:
            avversari.setdefault(o, [0, 0])
            avversari[o][1] += 1
            if win: avversari[o][0] += 1

    bp, bp_wr = "N/A", -1
    for p, s in compagni.items():
        if s[1] >= STATS_MIN_PAIR_PG:
            curr_wr = (s[0]/s[1])*100
            if curr_wr > bp_wr: bp, bp_wr = f"{p} ({curr_wr:.0f}%)", curr_wr
    
    nem, nem_wr = "N/A", 101
    for p, s in avversari.items():
        if s[1] >= STATS_MIN_PAIR_PG:
            curr_wr = (s[0]/s[1])*100
            if curr_wr < nem_wr: nem, nem_wr = f"{p} ({curr_wr:.0f}%)", curr_wr

    return {"wr": wr, "best_partner": bp, "nemesis": nem, "totale": totale}

def applica_decadimento(elo_attuale, data_ultima_partita, data_riferimento):
    if pd.isna(data_ultima_partita) or pd.isna(data_riferimento):
        return elo_attuale
    
    giorni_passati = (data_riferimento - data_ultima_partita).days
    
    if giorni_passati > DECAY_GIORNI_GRAZIA:
        giorni_over = giorni_passati - DECAY_GIORNI_GRAZIA
        malus = giorni_over * DECAY_PUNTI_GIORNO
        nuovo_elo = max(DECAY_MIN_ELO, elo_attuale - malus)
        return min(elo_attuale, nuovo_elo)
    return elo_attuale

def ricalcola_classifica():
    log = st.session_state.get('log_partite', pd.DataFrame(columns=COLONNE_LOG))
    current_players = st.session_state.get('lista_giocatori', [])
    
    # Raccogli TUTTI i giocatori unici mai apparsi nei log + quelli attivi
    all_known_players = set(current_players)
    if not log.empty:
        for p_list in log['giocatori']:
            if isinstance(p_list, list):
                all_known_players.update(p_list)
    
    # Inizializza strutture dati per TUTTI i giocatori noti
    # Questo previene il KeyError se un giocatore è nel log ma non nella lista attiva
    df = pd.DataFrame(0.0, index=list(all_known_players), columns=COLONNE_CLASSIFICA[1:])
    df['PG'], df['Elo'] = 0, ELO_STARTING
    df['UltimaPartita'] = pd.NaT 
    
    elo_hist = {p: [{'GameNum': 0, 'Elo': ELO_STARTING}] for p in all_known_players}
    p_games = {p: 0 for p in all_known_players}
    last_activity = {p: None for p in all_known_players}

    if log.empty or log.dropna(subset=['data']).empty:
        st.session_state.classifica = df.reset_index().rename(columns={'index': 'Giocatore'})
        st.session_state.elo_history = elo_hist
        return

    # MPP Logic
    log_val = log.dropna(subset=['giocatori', 'vincitori', 'data'])
    df['PG'].update(log_val['giocatori'].explode().value_counts())
    
    for row in log_val.itertuples():
        col_v = f'V{int(row.num_giocatori)}'
        if col_v in df.columns:
            if isinstance(row.vincitori, list):
                for v in row.vincitori:
                    if v in df.index:
                        df.loc[v, col_v] += 1
                        df.loc[v, 'PT'] += (row.punti_vittoria + row.punti_bonus)
    
    df['MPP'] = np.where(df['PG'] > 0, df['PT'] / df['PG'], 0).round(3)

    # Elo Logic
    log_elo = log_val.sort_values('data', ascending=True)
    curr_elo = df['Elo'].to_dict()
    
    start_date = log_elo.iloc[0]['data']
    # Inizializza data per tutti (evita NoneTypes)
    for p in all_known_players: last_activity[p] = start_date

    for row in log_elo.itertuples():
        match_date = row.data
        gs, vs, n = row.giocatori, row.vincitori, row.num_giocatori
        
        # Validazione base
        if not isinstance(gs, list) or not isinstance(vs, list): continue

        ps = [p for p in gs if p not in vs]
        gm = gs 
        
        # 1. Decadimento PRE-MATCH
        for p in gm:
            # Protezione aggiuntiva KeyError
            if p not in last_activity: last_activity[p] = start_date
            if p not in curr_elo: curr_elo[p] = ELO_STARTING

            if last_activity[p] is not None:
                curr_elo[p] = applica_decadimento(curr_elo[p], last_activity[p], match_date)
        
        # 2. Calcolo Elo Match
        try:
            if n == 2:
                w, l = vs[0], ps[0]
                rw, rl = curr_elo[w], curr_elo[l]
                e = 1 / (1 + 10**((rl - rw) / 400))
                d = ELO_K_FACTOR * (1 - e)
                curr_elo[w] += d; curr_elo[l] -= d
            elif n == 4:
                rw = (curr_elo[vs[0]] + curr_elo[vs[1]]) / 2
                rl = (curr_elo[ps[0]] + curr_elo[ps[1]]) / 2
                e = 1 / (1 + 10**((rl - rw) / 400))
                d = ELO_K_FACTOR * (1 - e)
                curr_elo[vs[0]] += d; curr_elo[vs[1]] += d
                curr_elo[ps[0]] -= d; curr_elo[ps[1]] -= d
            elif n == 3:
                w, l1, l2 = vs[0], ps[0], ps[1]
                rw = curr_elo[w]
                rl_avg = (curr_elo[l1] + curr_elo[l2]) / 2
                e = 1 / (1 + 10**((rl_avg - rw) / 400))
                d = ELO_K_FACTOR * (1 - e)
                curr_elo[w] += d
                curr_elo[l1] -= d/2; curr_elo[l2] -= d/2
            
            # 3. Aggiorna date e storico
            for p in gm:
                last_activity[p] = match_date
                p_games[p] += 1
                if p not in elo_hist: elo_hist[p] = [{'GameNum': 0, 'Elo': ELO_STARTING}]
                elo_hist[p].append({'GameNum': p_games[p], 'Elo': int(round(curr_elo[p]))})
        except Exception: 
            continue

    # 4. Decadimento FINALE (solo per i giocatori attivi o presenti)
    now = datetime.now()
    for p in all_known_players:
        if p in curr_elo and last_activity[p] is not None:
            curr_elo[p] = applica_decadimento(curr_elo[p], last_activity[p], now)
            df.loc[p, 'UltimaPartita'] = last_activity[p]

    df['Elo'] = df.index.map(curr_elo).fillna(ELO_STARTING).round(0).astype(int)
    st.session_state.classifica = df.reset_index().rename(columns={'index': 'Giocatore'})
    st.session_state.elo_history = elo_hist

def inizializza_stato():
    if 'log_caricato' not in st.session_state: st.session_state.log_caricato = False
    
    if 'gs_sh' not in st.session_state:
        st.session_state.gs_sh = connect_to_gsheet()
        if st.session_state.gs_sh:
            st.session_state.ws_log, st.session_state.ws_players = get_worksheets(st.session_state.gs_sh)
    
    if st.session_state.gs_sh and not st.session_state.log_caricato:
        log, lista = carica_dati(st.session_state.ws_log, st.session_state.ws_players)
        st.session_state.log_partite = log
        st.session_state.lista_giocatori = lista
        st.session_state.log_caricato = True
        ricalcola_classifica()

def registra_partita(gs, vs, bonus):
    row = pd.DataFrame([{
        "data": datetime.now(), "giorno_settimana": datetime.now().strftime('%A'),
        "giocatori": gs, "vincitori": vs, "num_giocatori": len(gs),
        "punti_vittoria": PUNTI_MAP[len(gs)], "punti_bonus": PUNTI_BONUS_100 if bonus else 0.0
    }])
    st.session_state.log_partite = pd.concat([st.session_state.log_partite, row], ignore_index=True)
    salva_log_gsheet(st.session_state.log_partite)
    ricalcola_classifica()
    st.toast("Match Saved!", icon="🎉")
    return True

def gestisci_giocatori_callback(action, name_val, old_name=None, pwd=None):
    if pwd != st.secrets["credentials"]["password"]:
        st.error("Wrong Password")
        return

    if action == "add":
        if name_val and name_val not in st.session_state.lista_giocatori:
            st.session_state.lista_giocatori.append(name_val)
            salva_giocatori_gsheet(st.session_state.lista_giocatori)
            st.success(f"Added {name_val}")
        else:
            st.warning("Name invalid or already exists")

    elif action == "rename":
        if old_name and name_val and name_val not in st.session_state.lista_giocatori:
            # 1. Update List
            idx = st.session_state.lista_giocatori.index(old_name)
            st.session_state.lista_giocatori[idx] = name_val
            salva_giocatori_gsheet(st.session_state.lista_giocatori)
            
            # 2. Update Log
            def replace_recursive(x):
                if isinstance(x, list):
                    return [name_val if i == old_name else i for i in x]
                return x
            
            st.session_state.log_partite['giocatori'] = st.session_state.log_partite['giocatori'].apply(replace_recursive)
            st.session_state.log_partite['vincitori'] = st.session_state.log_partite['vincitori'].apply(replace_recursive)
            
            salva_log_gsheet(st.session_state.log_partite)
            ricalcola_classifica()
            st.success(f"Renamed {old_name} -> {name_val} in all history.")
            st.rerun()

# --- 4. MAIN UI ---

def main():
    inizializza_stato()
    
    lista_attuale = st.session_state.get('lista_giocatori', [])

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 📝 Register Match")
        
        gs = st.multiselect("Select Players", lista_attuale, key="ms_giocatori", placeholder="Choose 2-4 players")
        vs = []
        n = len(gs)
        
        if n == 2 or n == 3:
            st.caption("Select Winner")
            w = st.radio("W", gs, index=None, label_visibility="collapsed")
            if w: vs = [w]
        elif n == 4:
            st.caption("Select 2 Winners")
            vs = st.multiselect("W", gs, max_selections=2, label_visibility="collapsed", placeholder="Select winning team")
        elif n > 0:
            st.caption("⚠️ Invalid number of players")

        bonus = st.checkbox(f"Bonus >100 (+{PUNTI_BONUS_100})", key="check_bonus")
        pwd = st.text_input("Admin Password", type="password", key="pwd_write")
        
        if st.button("💾 Save Match", use_container_width=True, type="primary"):
            if pwd == st.secrets["credentials"]["password"]:
                 if n in [2,3,4] and ((n<4 and len(vs)==1) or (n==4 and len(vs)==2)):
                     registra_partita(gs, vs, bonus)
                     st.session_state.ms_giocatori = []
                     st.rerun()
                 else:
                     st.error("Check winners/players count")
            else:
                st.error("Wrong Password")

        st.markdown("---")
        
        # --- PLAYER MANAGEMENT ---
        with st.expander("👥 Manage Players"):
            tab_add, tab_ren = st.tabs(["Add", "Rename"])
            
            with tab_add:
                new_p = st.text_input("New Player Name")
                if st.button("Add Player"):
                    gestisci_giocatori_callback("add", new_p, pwd=pwd)
            
            with tab_ren:
                p_ren = st.selectbox("Select Player", lista_attuale)
                new_n = st.text_input("Rename To")
                if st.button("Confirm Rename"):
                    gestisci_giocatori_callback("rename", new_n, old_name=p_ren, pwd=pwd)

        st.markdown("---")
        with st.expander("🗑️ Delete Last Match"):
            if st.button("Confirm Delete", use_container_width=True):
                if pwd == st.secrets["credentials"]["password"]:
                    if not st.session_state.log_partite.empty:
                        st.session_state.log_partite = st.session_state.log_partite.sort_values('data').iloc[:-1]
                        salva_log_gsheet(st.session_state.log_partite)
                        ricalcola_classifica()
                        st.success("Deleted!")
                        st.rerun()
                else: st.error("Wrong Password")

    # --- HEADER ---
    st.markdown('<div class="hero-title">♠️ Briscola League</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">Official Elo Rankings • Inactive players (> {DECAY_GIORNI_GRAZIA} days) are hidden</div>', unsafe_allow_html=True)

    df = st.session_state.classifica.copy()
    
    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "📈 Analytics", "📝 Log"])

    # --- TAB 1: LEADERBOARD ---
    with tab1:
        now = datetime.now()
        cutoff_date = now - timedelta(days=DECAY_GIORNI_GRAZIA)
        
        # Filtro Attivi
        mask_active = (df['UltimaPartita'] >= cutoff_date) | (df['UltimaPartita'].isna())
        # Filtra anche per essere nella lista attuale (esclude vecchi giocatori rinominati o rimossi)
        mask_in_list = df['Giocatore'].isin(lista_attuale)
        
        df_active = df[mask_active & mask_in_list].copy()

        # PODIUM
        podio = df_active[df_active['PG'] >= PODIO_MIN_PG].sort_values(by=["Elo", "PG"], ascending=[False, True]).reset_index(drop=True)
        
        if not podio.empty:
            cols = st.columns(3)
            medals = [("🥇", "gold"), ("🥈", "silver"), ("🥉", "bronze")]
            for i in range(min(3, len(podio))):
                p = podio.iloc[i]
                with cols[i]:
                    st.markdown(f"""
                    <div class="podium-card {medals[i][1]}">
                        <div class="medal">{medals[i][0]}</div>
                        <div class="player-name">{p['Giocatore']}</div>
                        <div class="player-elo">{p['Elo']} Elo</div>
                        <div class="player-stats">{int(p['PG'])} matches</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"No active players have reached {PODIO_MIN_PG} matches.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # TABLE
        col_filter, _ = st.columns([1, 2])
        with col_filter:
            max_pg_slider = int(df_active['PG'].max()) if not df_active.empty else 0
            min_pg = st.slider("Filter by Min. Matches", 0, max_pg_slider, 2)
        
        df_show = df_active[df_active['PG'] >= min_pg].sort_values(["Elo", "PG"], ascending=[False, True])
        df_show.insert(0, 'Rank', range(1, 1 + len(df_show)))

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            column_order=["Rank", "Giocatore", "Elo", "PG", "PT", "MPP", "UltimaPartita"],
            column_config={
                "Rank": st.column_config.NumberColumn("#", format="%d", width="small"),
                "Giocatore": st.column_config.TextColumn("Player", width="medium"),
                "Elo": st.column_config.ProgressColumn("Elo Rating", format="%d", min_value=800, max_value=1300),
                "PG": st.column_config.NumberColumn("Matches", format="%d"),
                "UltimaPartita": st.column_config.DateColumn("Last Match", format="DD/MM/YYYY")
            }
        )

    # --- TAB 2: ANALYTICS ---
    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📈 Elo History")
            top_pl = df_active.sort_values("Elo", ascending=False).head(5)['Giocatore'].tolist()
            sel_pl = st.multiselect("Compare Players", lista_attuale, default=top_pl)
            
            if sel_pl:
                all_data = []
                for p in sel_pl:
                    if p in st.session_state.elo_history:
                        for r in st.session_state.elo_history[p]:
                            all_data.append({'Player': p, 'Match': r['GameNum'], 'Elo': r['Elo']})
                
                if all_data:
                    df_chart = pd.DataFrame(all_data)
                    ymin, ymax = (df_chart['Elo'].min()-20, df_chart['Elo'].max()+20)
                    chart = alt.Chart(df_chart).mark_line(point=True, strokeWidth=3).encode(
                        x=alt.X('Match', title='Matches Played'),
                        y=alt.Y('Elo', scale=alt.Scale(domain=[ymin, ymax]), title='Elo Rating'),
                        color='Player',
                        tooltip=['Player', 'Match', 'Elo']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

        with c2:
            st.markdown("### 🕵️ Player Insights")
            p_sort = df_active.sort_values("Elo", ascending=False)['Giocatore'].tolist()
            for p in p_sort:
                stats = calcola_stats_dettagliate(p, st.session_state.log_partite)
                if stats:
                    with st.expander(f"**{p}**"):
                        st.write(f"**Win Rate:** {stats['wr']:.1f}% ({stats['totale']} games)")
                        st.write(f"**🤝 Best Partner:** {stats['best_partner']}")
                        st.write(f"**💀 Nemesis:** {stats['nemesis']}")

    # --- TAB 3: LOG ---
    with tab3:
        if not st.session_state.log_partite.empty:
            st.dataframe(
                st.session_state.log_partite.sort_values('data', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "data": st.column_config.DatetimeColumn("Timestamp", format="DD/MM/YY HH:mm"),
                    "vincitori": st.column_config.ListColumn("Winners"),
                    "giocatori": st.column_config.ListColumn("Players")
                }
            )

if __name__ == "__main__":
    main()