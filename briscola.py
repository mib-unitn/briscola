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

LISTA_GIOCATORI = ["Michele", "Federico", "Lorenza", "Grazia", "Pierpaolo", "Niccolò", "Simone", "Avinash", "Sahil", "Massine", "Beppe", "Guest", "Giampaolo"]

PUNTI_MAP = {2: 2, 3: 3, 4: 2}
PUNTI_BONUS_100 = 0.5

ELO_STARTING = 1000
ELO_K_FACTOR = 32
PODIO_MIN_PG = 40
STATS_MIN_TOTAL_PG = 25
STATS_MIN_PAIR_PG = 10

# --- CONFIGURAZIONE DECADIMENTO (DECAY) ---
DECAY_GIORNI_GRAZIA = 10  # Giorni di inattività prima di essere nascosti/penalizzati
DECAY_PUNTI_GIORNO = 5    # Punti persi al giorno
DECAY_MIN_ELO = 750       # Floor Elo

COLONNE_LOG = ["data", "giorno_settimana", "giocatori", "vincitori", "num_giocatori", "punti_vittoria", "punti_bonus"]
COLONNE_CLASSIFICA = ["Giocatore", "PG", "V2", "V3", "V4", "PT", "MPP", "Elo", "UltimaPartita"]

# --- 2. CUSTOM CSS & STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Title Styling */
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

    /* Custom Podium Cards */
    .podium-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
        flex-wrap: wrap;
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

    /* Gold Styling */
    .gold { border-top: 6px solid #FFD700; background: linear-gradient(180deg, #fff 0%, #fffdf0 100%); }
    /* Silver Styling */
    .silver { border-top: 6px solid #C0C0C0; background: linear-gradient(180deg, #fff 0%, #f8f9fa 100%); }
    /* Bronze Styling */
    .bronze { border-top: 6px solid #CD7F32; background: linear-gradient(180deg, #fff 0%, #fff5f0 100%); }
    
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND FUNCTIONS ---

def connect_to_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open_by_url(GOOGLE_SHEET_URL).sheet1
    except Exception as e:
        st.error(f"GSheets Connection Error: {e}")
        return None

def carica_log_gsheet(worksheet):
    try:
        df = get_as_dataframe(worksheet, evaluate_formulas=True, dtype_backend='pyarrow')
        if df.empty: return pd.DataFrame(columns=COLONNE_LOG)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        for col in ['giocatori', 'vincitori']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        
        for c in ['num_giocatori', 'punti_vittoria', 'punti_bonus']:
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
            df_save = df.copy()
            df_save['giocatori'] = df_save['giocatori'].astype(str)
            df_save['vincitori'] = df_save['vincitori'].astype(str)
            st.session_state.gs_worksheet.clear()
            set_with_dataframe(st.session_state.gs_worksheet, df_save, include_index=False, resize=True)
            return True
        except: return False
    return False

def calcola_stats_dettagliate(player, log):
    df_p = log[log['giocatori'].apply(lambda x: player in x)]
    if df_p.empty: return None

    totale = len(df_p)
    vittorie = len(df_p[df_p['vincitori'].apply(lambda x: player in x)])
    wr = (vittorie / totale) * 100

    if totale < STATS_MIN_TOTAL_PG:
        msg = "Not enough data"
        return {"wr": wr, "best_partner": msg, "nemesis": msg, "totale": totale}

    compagni, avversari = {}, {}

    for row in df_p.itertuples():
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

    # Best Partner
    bp, bp_wr = "N/A", -1
    for p, s in compagni.items():
        if s[1] >= STATS_MIN_PAIR_PG:
            curr_wr = (s[0]/s[1])*100
            if curr_wr > bp_wr: bp, bp_wr = f"{p} ({curr_wr:.0f}%)", curr_wr
    
    # Nemesis
    nem, nem_wr = "N/A", 101
    for p, s in avversari.items():
        if s[1] >= STATS_MIN_PAIR_PG:
            curr_wr = (s[0]/s[1])*100
            if curr_wr < nem_wr: nem, nem_wr = f"{p} ({curr_wr:.0f}%)", curr_wr

    return {"wr": wr, "best_partner": bp, "nemesis": nem, "totale": totale}

def applica_decadimento(elo_attuale, data_ultima_partita, data_riferimento):
    """Calcola il nuovo Elo applicando il decadimento basato sul tempo."""
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
    df = pd.DataFrame(0.0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:])
    df['PG'], df['Elo'] = 0, ELO_STARTING
    df['UltimaPartita'] = pd.NaT 
    
    elo_hist = {p: [{'GameNum': 0, 'Elo': ELO_STARTING}] for p in LISTA_GIOCATORI}
    p_games = {p: 0 for p in LISTA_GIOCATORI}
    
    last_activity = {p: None for p in LISTA_GIOCATORI}

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
            for v in row.vincitori:
                if v in df.index:
                    df.loc[v, col_v] += 1
                    df.loc[v, 'PT'] += (row.punti_vittoria + row.punti_bonus)
    
    df['MPP'] = np.where(df['PG'] > 0, df['PT'] / df['PG'], 0).round(3)

    # Elo Logic con Decadimento
    log_elo = log_val.sort_values('data', ascending=True)
    curr_elo = df['Elo'].to_dict()
    
    start_date = log_elo.iloc[0]['data']
    for p in LISTA_GIOCATORI: last_activity[p] = start_date

    for row in log_elo.itertuples():
        match_date = row.data
        gs, vs, n = row.giocatori, row.vincitori, row.num_giocatori
        ps = [p for p in gs if p not in vs]
        gm = gs 
        
        # 1. Decadimento PRE-MATCH
        for p in gm:
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
                elo_hist[p].append({'GameNum': p_games[p], 'Elo': int(round(curr_elo[p]))})
        except: continue

    # 4. Decadimento FINALE (fino a Oggi)
    now = datetime.now()
    for p in LISTA_GIOCATORI:
        if last_activity[p] is not None:
            curr_elo[p] = applica_decadimento(curr_elo[p], last_activity[p], now)
            df.loc[p, 'UltimaPartita'] = last_activity[p]

    df['Elo'] = df.index.map(curr_elo).round(0).astype(int)
    st.session_state.classifica = df.reset_index().rename(columns={'index': 'Giocatore'})
    st.session_state.elo_history = elo_hist

def inizializza_stato():
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
    row = pd.DataFrame([{
        "data": datetime.now(), "giorno_settimana": datetime.now().strftime('%A'),
        "giocatori": gs, "vincitori": vs, "num_giocatori": len(gs),
        "punti_vittoria": PUNTI_MAP[len(gs)], "punti_bonus": PUNTI_BONUS_100 if bonus else 0.0
    }])
    st.session_state.log_partite = pd.concat([st.session_state.log_partite, row], ignore_index=True)
    salva_log_gsheet(st.session_state.log_partite)
    ricalcola_classifica()
    st.toast("Match Saved Successfully!", icon="🎉")
    return True

def callback_salva(gs, vs, bonus, pwd):
    try:
        if pwd != st.secrets["credentials"]["password"]:
            st.session_state['temp_err'] = "⛔ Wrong Password"
            return
    except:
        st.session_state['temp_err'] = "Secret Error"
        return

    n = len(gs)
    err = None
    if n not in [2,3,4]: err = "Select 2, 3 or 4 players"
    elif (n in [2,3] and len(vs)!=1) or (n==4 and len(vs)!=2): err = "Incorrect number of winners"
    
    if err:
        st.session_state['temp_err'] = err
        return

    if registra_partita(gs, vs, bonus):
        st.session_state.ms_giocatori = []
        st.session_state.check_bonus = False
        st.session_state['temp_err'] = None

# --- 4. MAIN UI ---

def main():
    inizializza_stato()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 📝 Register Match")
        
        if 'temp_err' in st.session_state and st.session_state['temp_err']:
            st.error(st.session_state['temp_err'])
            st.session_state['temp_err'] = None 
        
        gs = st.multiselect("Select Players", LISTA_GIOCATORI, key="ms_giocatori", placeholder="Choose 2-4 players")
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
        st.markdown("---")
        pwd = st.text_input("Admin Password", type="password", key="pwd_write")
        
        st.button("💾 Save Match", use_container_width=True, type="primary", on_click=callback_salva, args=(gs, vs, bonus, pwd))

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
                    else: st.error("Log empty")
                else: st.error("Wrong Password")

    # --- HEADER ---
    st.markdown('<div class="hero-title">♠️ Briscola League</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">Official Elo Rankings • Inactive players (> {DECAY_GIORNI_GRAZIA} days) are hidden</div>', unsafe_allow_html=True)

    df = st.session_state.classifica.copy()
    
    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "📈 Analytics", "📝 Log"])

    # --- TAB 1: LEADERBOARD ---
    with tab1:
        # FILTER ACTIVE PLAYERS
        now = datetime.now()
        cutoff_date = now - timedelta(days=DECAY_GIORNI_GRAZIA)
        
        # Filtra: Mantieni chi ha giocato recentemente (>= cutoff) OPPURE chi non ha mai giocato (NaT)
        # Nota: Chi ha NaT (0 partite) viene solitamente escluso dal filtro PG dopo.
        mask_active = (df['UltimaPartita'] >= cutoff_date) | (df['UltimaPartita'].isna())
        df_active = df[mask_active].copy()

        # PODIUM HTML (su df_active)
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
            st.info(f"No active players have reached {PODIO_MIN_PG} matches for the Elite Podium.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # TABLE (su df_active)
        st.markdown("### 📊 Active Players")
        col_filter, col_spacer = st.columns([1, 2])
        with col_filter:
            # Slider max su df_active per coerenza
            max_pg_slider = int(df_active['PG'].max()) if not df_active.empty else 0
            min_pg = st.slider("Filter by Min. Matches", 0, max_pg_slider, 2)
        
        df_show = df_active[df_active['PG'] >= min_pg].sort_values(["Elo", "PG"], ascending=[False, True])
        
        # Add Rank
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
                "PT": st.column_config.NumberColumn("Points", format="%.1f"),
                "MPP": st.column_config.NumberColumn("Avg Pts", format="%.2f"),
                "UltimaPartita": st.column_config.DateColumn("Last Match", format="DD/MM/YYYY")
            }
        )

    # --- TAB 2: ANALYTICS ---
    with tab2:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("### 📈 Elo History")
            # Top players selection from ALL players (history is history)
            top_pl = df.sort_values("Elo", ascending=False).head(5)['Giocatore'].tolist()
            sel_pl = st.multiselect("Compare Players", LISTA_GIOCATORI, default=top_pl)
            
            if sel_pl:
                all_data = []
                all_vals = []
                for p in st.session_state.elo_history:
                    for r in st.session_state.elo_history[p]: all_vals.append(r['Elo'])
                ymin, ymax = (min(all_vals)-20, max(all_vals)+20) if all_vals else (900, 1100)

                for p in sel_pl:
                    if p in st.session_state.elo_history:
                        for r in st.session_state.elo_history[p]:
                            all_data.append({'Player': p, 'Match': r['GameNum'], 'Elo': r['Elo']})
                
                if all_data:
                    chart = alt.Chart(pd.DataFrame(all_data)).mark_line(point=True, strokeWidth=3).encode(
                        x=alt.X('Match', title='Matches Played'),
                        y=alt.Y('Elo', scale=alt.Scale(domain=[ymin, ymax]), title='Elo Rating'),
                        color='Player',
                        tooltip=['Player', 'Match', 'Elo']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

        with c2:
            st.markdown("### 🕵️ Player Insights")
            st.caption(f"Min. {STATS_MIN_TOTAL_PG} total matches / {STATS_MIN_PAIR_PG} pair matches")
            
            p_sort = df.sort_values("Elo", ascending=False)['Giocatore'].tolist()
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
                    "giocatori": st.column_config.ListColumn("Players"),
                    "punti_vittoria": st.column_config.NumberColumn("Pts"),
                    "punti_bonus": st.column_config.NumberColumn("Bonus")
                }
            )
        else:
            st.info("No matches recorded yet.")

if __name__ == "__main__":
    main()