import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import ast
import altair as alt

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Torneo Briscola", 
    layout="wide", 
    page_icon="🃏",
    initial_sidebar_state="expanded"
)

# --- 1. Configurazione Variabili ---

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ea2DT-FlYmq_TjA4MFqdc1rr3GFj1MxpVfWm0MQxrQo/edit?gid=0#gid=0"

LISTA_GIOCATORI = ["Michele", "Federico", "Lorenza", "Grazia", "Pierpaolo", "Niccolò", "Simone", "Avinash", "Sahil", "Massine", "Beppe", "Esteban", "Giampaolo"]

PUNTI_MAP = {2: 2, 3: 3, 4: 2}
PUNTI_BONUS_100 = 0.5

ELO_STARTING = 1000
ELO_K_FACTOR = 32
PODIO_MIN_PG = 20

COLONNE_LOG = ["data", "giorno_settimana", "giocatori", "vincitori", "num_giocatori", "punti_vittoria", "punti_bonus"]
COLONNE_CLASSIFICA = ["Giocatore", "PG", "V2", "V3", "V4", "PT", "MPP", "Elo"]

# --- CSS PERSONALIZZATO ---
st.markdown("""
    <style>
    .main-title { font-size: 3rem !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: 0px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .subtitle { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; font-style: italic; }
    div[data-testid="metric-container"] { background-color: #f0f2f6; border-radius: 10px; padding: 15px; border: 1px solid #dcdcdc; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); transition: transform 0.2s; }
    div[data-testid="metric-container"]:hover { transform: scale(1.02); }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="metric-container"] { border-left: 5px solid #FFD700; }
    div[data-testid="column"]:nth-of-type(2) div[data-testid="metric-container"] { border-left: 5px solid #C0C0C0; }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="metric-container"] { border-left: 5px solid #CD7F32; }
    section[data-testid="stSidebar"] { background-color: #f9f9f9; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Funzioni Backend ---

def connect_to_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        worksheet = gc.open_by_url(GOOGLE_SHEET_URL).sheet1
        return worksheet
    except Exception as e:
        st.error(f"Errore GSheets: {e}")
        return None

def carica_log_gsheet(worksheet):
    try:
        df = get_as_dataframe(worksheet, evaluate_formulas=True, dtype_backend='pyarrow')
        if df.empty: return pd.DataFrame(columns=COLONNE_LOG)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.empty: return pd.DataFrame(columns=COLONNE_LOG)
        
        if 'giocatori' in df.columns:
            df['giocatori'] = df['giocatori'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        if 'vincitori' in df.columns:
            df['vincitori'] = df['vincitori'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        
        cols_num = ['num_giocatori', 'punti_vittoria', 'punti_bonus']
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
        else:
            df['data'] = pd.NaT

        for col in COLONNE_LOG:
            if col not in df.columns: df[col] = pd.NA
            
        return df[COLONNE_LOG].astype({'punti_bonus': 'float', 'punti_vittoria': 'float', 'num_giocatori': 'int'})
    except Exception:
        return pd.DataFrame(columns=COLONNE_LOG)

def salva_log_gsheet(df):
    if 'gs_worksheet' in st.session_state and st.session_state.gs_worksheet:
        try:
            df_to_save = df.copy()
            df_to_save['giocatori'] = df_to_save['giocatori'].astype(str)
            df_to_save['vincitori'] = df_to_save['vincitori'].astype(str)
            st.session_state.gs_worksheet.clear()
            set_with_dataframe(st.session_state.gs_worksheet, df_to_save, include_index=False, resize=True)
            return True
        except:
            return False
    return False

def calcola_stats_dettagliate(player, log):
    """Calcola statistiche avanzate per singolo giocatore."""
    # Filtra partite dove il giocatore ha partecipato
    df_p = log[log['giocatori'].apply(lambda x: player in x)]
    
    if df_p.empty:
        return None

    totale_partite = len(df_p)
    vittorie = len(df_p[df_p['vincitori'].apply(lambda x: player in x)])
    win_rate = (vittorie / totale_partite) * 100

    # Dizionari per contatori
    compagni = {} # {nome: [vittorie_insieme, totale_insieme]}
    avversari = {} # {nome: [vittorie_contro, totale_contro]}

    for row in df_p.itertuples():
        ha_vinto = player in row.vincitori
        
        # Logica Compagni (Solo 2v2)
        if row.num_giocatori == 4:
            # Trova chi era in squadra con me
            my_team = row.vincitori if ha_vinto else [p for p in row.giocatori if p not in row.vincitori]
            # Il compagno è colui che è nel mio team ma non sono io
            partners = [p for p in my_team if p != player]
            if partners:
                partner = partners[0]
                if partner not in compagni: compagni[partner] = [0, 0]
                compagni[partner][1] += 1 # Incrementa partite totali
                if ha_vinto: compagni[partner][0] += 1 # Incrementa vittorie

        # Logica Avversari (Tutte le modalità)
        # Se ho vinto, gli avversari sono i perdenti. Se ho perso, sono i vincitori.
        opponents = [p for p in row.giocatori if p not in row.vincitori] if ha_vinto else row.vincitori
        for opp in opponents:
            if opp not in avversari: avversari[opp] = [0, 0]
            avversari[opp][1] += 1 # Totale contro
            if ha_vinto: avversari[opp][0] += 1 # Vittorie contro

    # Trova Miglior Compagno (Minimo 3 partite insieme)
    best_partner, best_partner_wr = "N/A", 0
    for p, stats in compagni.items():
        if stats[1] >= 3:
            wr = (stats[0] / stats[1]) * 100
            if wr > best_partner_wr:
                best_partner_wr = wr
                best_partner = f"{p} ({wr:.0f}%)"

    # Trova Bestia Nera (Peggior Winrate contro, Minimo 3 partite)
    nemesis, nemesis_wr = "N/A", 101
    for p, stats in avversari.items():
        if stats[1] >= 3:
            wr = (stats[0] / stats[1]) * 100
            if wr < nemesis_wr:
                nemesis_wr = wr
                nemesis = f"{p} ({wr:.0f}%)"
    
    # Trova Cliente (Miglior Winrate contro, Minimo 3 partite)
    pigeon, pigeon_wr = "N/A", -1
    for p, stats in avversari.items():
        if stats[1] >= 3:
            wr = (stats[0] / stats[1]) * 100
            if wr > pigeon_wr:
                pigeon_wr = wr
                pigeon = f"{p} ({wr:.0f}%)"

    return {
        "wr": win_rate,
        "best_partner": best_partner,
        "nemesis": nemesis, # Contro chi perdi sempre
        "pigeon": pigeon    # Contro chi vinci sempre
    }

def ricalcola_classifica():
    log = st.session_state.get('log_partite', pd.DataFrame(columns=COLONNE_LOG))
    
    classifica_nuova = pd.DataFrame(0.0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:])
    classifica_nuova['PG'] = 0
    classifica_nuova['Elo'] = ELO_STARTING
    
    elo_history_dict = {p: [{'GameNum': 0, 'Elo': ELO_STARTING}] for p in LISTA_GIOCATORI}
    player_game_counts = {p: 0 for p in LISTA_GIOCATORI}

    if log.empty or log.dropna(subset=['data']).empty:
        st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})
        st.session_state.elo_history = elo_history_dict
        return

    # MPP (Backend)
    log_valido = log.dropna(subset=['giocatori', 'vincitori', 'data'])
    classifica_nuova['PG'].update(log_valido['giocatori'].explode().value_counts())
    
    for partita in log_valido.itertuples():
        col_vittoria = f'V{int(partita.num_giocatori)}'
        if col_vittoria in classifica_nuova.columns:
            for vincitore in partita.vincitori:
                if vincitore in classifica_nuova.index:
                    classifica_nuova.loc[vincitore, col_vittoria] += 1
                    classifica_nuova.loc[vincitore, 'PT'] += (partita.punti_vittoria + partita.punti_bonus)
    
    classifica_nuova['MPP'] = np.where(classifica_nuova['PG'] > 0, classifica_nuova['PT'] / classifica_nuova['PG'], 0).round(3)

    # Elo
    log_elo = log_valido.sort_values(by="data", ascending=True)
    elo_curr = classifica_nuova['Elo'].to_dict()
    
    for partita in log_elo.itertuples():
        gs = partita.giocatori
        vs = partita.vincitori
        ps = [p for p in gs if p not in vs]
        n = partita.num_giocatori
        gm = []

        try:
            if n == 2:
                w, l = vs[0], ps[0]
                rw, rl = elo_curr[w], elo_curr[l]
                ew = 1 / (1 + 10**((rl - rw) / 400))
                d = ELO_K_FACTOR * (1 - ew)
                elo_curr[w] += d; elo_curr[l] -= d
                gm = [w, l]
            elif n == 4:
                rw = (elo_curr[vs[0]] + elo_curr[vs[1]]) / 2
                rl = (elo_curr[ps[0]] + elo_curr[ps[1]]) / 2
                ew = 1 / (1 + 10**((rl - rw) / 400))
                d = ELO_K_FACTOR * (1 - ew)
                elo_curr[vs[0]] += d; elo_curr[vs[1]] += d
                elo_curr[ps[0]] -= d; elo_curr[ps[1]] -= d
                gm = vs + ps
            elif n == 3:
                w, l1, l2 = vs[0], ps[0], ps[1]
                rw = elo_curr[w]
                rl_avg = (elo_curr[l1] + elo_curr[l2]) / 2
                ew = 1 / (1 + 10**((rl_avg - rw) / 400))
                d = ELO_K_FACTOR * (1 - ew)
                elo_curr[w] += d
                elo_curr[l1] -= d/2; elo_curr[l2] -= d/2
                gm = [w, l1, l2]
            
            for p in gm:
                player_game_counts[p] += 1
                elo_history_dict[p].append({'GameNum': player_game_counts[p], 'Elo': int(round(elo_curr[p]))})
                
        except: continue
        
    classifica_nuova['Elo'] = classifica_nuova.index.map(elo_curr).round(0).astype(int)
    st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})
    st.session_state.elo_history = elo_history_dict

def inizializza_stato():
    if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
    if 'log_partite' not in st.session_state: st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
    if 'classifica' not in st.session_state:
        df = pd.DataFrame(0.0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:])
        df['PG'], df['Elo'] = 0, ELO_STARTING
        st.session_state.classifica = df.reset_index().rename(columns={'index': 'Giocatore'})
    if 'elo_history' not in st.session_state: st.session_state.elo_history = {}
    if 'gs_worksheet' not in st.session_state: st.session_state.gs_worksheet = None
    if 'log_caricato' not in st.session_state: st.session_state.log_caricato = False

    if st.session_state.gs_worksheet is None: st.session_state.gs_worksheet = connect_to_gsheet()
    if st.session_state.gs_worksheet and not st.session_state.log_caricato:
        st.session_state.log_partite = carica_log_gsheet(st.session_state.gs_worksheet)
        st.session_state.log_caricato = True
        ricalcola_classifica()

def registra_partita(gs, vs, bonus):
    n = len(gs)
    nv = len(vs)
    if not gs: return st.sidebar.error("Seleziona giocatori")
    if n < 2: return st.sidebar.error("Minimo 2 giocatori")
    if not vs: return st.sidebar.error("Seleziona vincitore")
    if (n in [2,3] and nv!=1) or (n==4 and nv!=2): return st.sidebar.error("Numero vincitori errato")
    for v in vs: 
        if v not in gs: return st.sidebar.error(f"{v} non ha giocato")
        
    row = pd.DataFrame([{
        "data": datetime.now(), "giorno_settimana": datetime.now().strftime('%A'),
        "giocatori": gs, "vincitori": vs, "num_giocatori": n,
        "punti_vittoria": PUNTI_MAP[n], "punti_bonus": PUNTI_BONUS_100 if bonus else 0.0
    }])
    st.session_state.log_partite = pd.concat([st.session_state.log_partite, row], ignore_index=True)
    salva_log_gsheet(st.session_state.log_partite)
    ricalcola_classifica()
    st.toast(f"Vittoria registrata per {', '.join(vs)}!", icon="✅")
    return True

def check_password():
    if st.session_state.get("password_correct", False): return True
    st.markdown("<h1 style='text-align: center;'>🔒 Accesso Protetto</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        pwd = st.text_input("Inserisci Password", type="password")
        if st.button("Entra", use_container_width=True):
            if pwd == st.secrets["credentials"]["password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else: st.error("Password Errata")
    return False

# --- 3. UI Principale ---

def main():
    inizializza_stato()
    if not check_password(): st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("📝 Registra")
        with st.form("registra_form", clear_on_submit=True):
            gs = st.multiselect("Giocatori", LISTA_GIOCATORI)
            vs = st.multiselect("Vincitori", gs)
            bonus = st.checkbox(f"Bonus >100 (+{PUNTI_BONUS_100})")
            submitted = st.form_submit_button("💾 Salva Partita", use_container_width=True, type="primary")
            
            if submitted:
                registra_partita(gs, vs, bonus)
        
        st.markdown("---")
        with st.expander("⚙️ Amministrazione"):
            if st.button("🗑️ Elimina Ultima", use_container_width=True):
                if not st.session_state.log_partite.empty:
                    st.session_state.log_partite = st.session_state.log_partite.sort_values('data').iloc[:-1]
                    salva_log_gsheet(st.session_state.log_partite)
                    ricalcola_classifica()
                    st.rerun()
            if st.button("🔥 Reset Totale", use_container_width=True):
                st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
                salva_log_gsheet(st.session_state.log_partite)
                st.session_state.elo_history = {}
                ricalcola_classifica()
                st.rerun()

    # --- MAIN PAGE ---
    st.markdown('<div class="main-title">🃏 Torneo Briscola</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Sistema Elo (Start: {ELO_STARTING}, K: {ELO_K_FACTOR}) - Podio Elite: {PODIO_MIN_PG}+ Partite</div>', unsafe_allow_html=True)

    df = st.session_state.classifica.copy()
    max_pg = int(df['PG'].max()) if not df.empty else 0

    # --- PODIO ---
    podio = df[df['PG'] >= PODIO_MIN_PG].sort_values(by=["Elo", "PG"], ascending=[False, True]).reset_index(drop=True)
    
    col1, col2, col3 = st.columns(3)
    if not podio.empty:
        if len(podio) > 0: col1.metric("🥇 1° Posto", podio.iloc[0]['Giocatore'], f"{podio.iloc[0]['Elo']} Elo")
        if len(podio) > 1: col2.metric("🥈 2° Posto", podio.iloc[1]['Giocatore'], f"{podio.iloc[1]['Elo']} Elo")
        if len(podio) > 2: col3.metric("🥉 3° Posto", podio.iloc[2]['Giocatore'], f"{podio.iloc[2]['Elo']} Elo")
    else:
        st.info(f"Nessun giocatore ha raggiunto {PODIO_MIN_PG} partite.")

    st.markdown("---")

    # --- GRAFICO ---
    st.subheader("📈 Andamento Elo")
    top3 = df.sort_values("Elo", ascending=False).head(3)['Giocatore'].tolist()
    sel_players = st.multiselect("Confronta giocatori", LISTA_GIOCATORI, default=top3)
    
    if sel_players:
        all_data, all_elos = [], []
        for p in st.session_state.elo_history:
             for r in st.session_state.elo_history[p]: all_elos.append(r['Elo'])
        
        ymin, ymax = (min(all_elos)-20, max(all_elos)+20) if all_elos else (900, 1100)

        for p in sel_players:
            if p in st.session_state.elo_history:
                for r in st.session_state.elo_history[p]:
                    all_data.append({'Giocatore': p, 'Partita': r['GameNum'], 'Elo': r['Elo']})
        
        if all_data:
            chart = alt.Chart(pd.DataFrame(all_data)).mark_line(point=True).encode(
                x=alt.X('Partita', title='Partite Giocate'),
                y=alt.Y('Elo', scale=alt.Scale(domain=[ymin, ymax])),
                color='Giocatore',
                tooltip=['Giocatore', 'Partita', 'Elo']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # --- NUOVA SEZIONE: STATISTICHE DETTAGLIATE GIOCATORI ---
    st.subheader("🕵️ Statistiche Giocatori")
    st.markdown("Clicca sul nome per vedere il winrate e le affinità.")
    
    # Ordina i giocatori per Elo per l'elenco
    players_sorted = df.sort_values(by="Elo", ascending=False)['Giocatore'].tolist()
    
    for p in players_sorted:
        # Recupera l'Elo corrente
        curr_elo = df[df['Giocatore'] == p]['Elo'].values[0]
        
        # Calcola le stats
        stats = calcola_stats_dettagliate(p, st.session_state.log_partite)
        
        if stats:
            with st.expander(f"**{p}** (Elo: {curr_elo} - Winrate: {stats['wr']:.1f}%)"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Win Rate Globale", f"{stats['wr']:.1f}%")
                c2.metric("🤝 Miglior Compagno (2v2)", stats['best_partner'], help="Con chi vinci di più quando siete in squadra insieme (Min 3 partite).")
                c3.metric("💀 Bestia Nera (Avversario)", stats['nemesis'], help="L'avversario contro cui hai la % di vittoria più bassa (Min 3 partite).")
                
                # c4.metric("💰 Cliente (Avversario)", stats['pigeon'], help="L'avversario contro cui hai la % di vittoria più alta (Min 3 partite).")

    st.markdown("---")

    # --- CLASSIFICA TABELLARE ---
    st.markdown("### 📊 Classifica Completa")
    min_pg_filter = st.slider("Filtra per Partite Minime", 0, max_pg+1, 2)
    df_show = df[df['PG'] >= min_pg_filter].sort_values(["Elo", "PG"], ascending=[False, True])
    
    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_order=["Giocatore", "Elo", "PG", "PT", "V2", "V3", "V4"],
        column_config={
            "Elo": st.column_config.ProgressColumn("Rating Elo", format="%d", min_value=800, max_value=1200),
            "PG": st.column_config.NumberColumn("Partite", format="%d"),
            "PT": st.column_config.NumberColumn("Punti Totali", format="%.1f"),
            "V2": st.column_config.NumberColumn("Vittorie (2)", format="%d"),
            "V3": st.column_config.NumberColumn("Vittorie (3)", format="%d"),
            "V4": st.column_config.NumberColumn("Vittorie (4)", format="%d"),
        }
    )

    # --- LOG ---
    with st.expander("📜 Storico Partite"):
        if not st.session_state.log_partite.empty:
            st.dataframe(
                st.session_state.log_partite.sort_values('data', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={"data": st.column_config.DatetimeColumn("Data", format="DD/MM HH:mm")}
            )

if __name__ == "__main__":
    main()