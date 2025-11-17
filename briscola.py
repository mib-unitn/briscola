import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import ast

# --- 1. Configurazione Iniziale ---

# INCOLLA QUI L'URL DEL TUO FOGLIO GOOGLE
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ea2DT-FlYmq_TjA4MFqdc1rr3GFj1MxpVfWm0MQxrQo/edit?gid=0#gid=0" # INCOLLA IL TUO URL

# !!! MODIFICA QUI I TUOI GIOCATORI !!!
LISTA_GIOCATORI = ["Michele", "Federico", "Lorenza", "Grazia", "Pierpaolo", "Niccolò", "Simone", "Avinash", "Sahil", "Massine", "Beppe", "Esteban"]

# --- MODIFICA: PUNTI BILANCIATI PER IL SISTEMA MPP ---
# Come discusso: +2 per 1v1, +2 per 2v2, +3 per 1v1v1
# Punti bilanciati per il sistema MPP
PUNTI_MAP = {
    2: 2,  # Vittoria in partita a 2
    3: 3,  # Vittoria in partita a 3
    4: 2   # Vittoria in partita a 4 (per ciascun vincitore)
}
PUNTI_BONUS_100 = 0.5

# Costanti Elo
ELO_STARTING = 1000
ELO_K_FACTOR = 32

# Colonne
COLONNE_LOG = ["data", "giorno_settimana", "giocatori", "vincitori", "num_giocatori", "punti_vittoria", "punti_bonus"]
COLONNE_CLASSIFICA = ["Giocatore", "PG", "V2", "V3", "V4", "PT", "MPP", "Elo"]

# --- 2. Funzioni di Persistenza Google Sheets ---

def connect_to_gsheet():
    """Connettiti a Google Sheets usando i secrets di Streamlit."""
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        worksheet = gc.open_by_url(GOOGLE_SHEET_URL).sheet1
        return worksheet
    except Exception as e:
        st.error(f"Errore di connessione a Google Sheets: {e}")
        return None

def carica_log_gsheet(worksheet):
    """Carica il log dal Foglio Google."""
    try:
        df = get_as_dataframe(worksheet, evaluate_formulas=True, dtype_backend='pyarrow')
        if df.empty:
            df = pd.DataFrame(columns=COLONNE_LOG)
        
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.empty:
             df = pd.DataFrame(columns=COLONNE_LOG)
             
        if 'giocatori' in df.columns and not df['giocatori'].empty:
            df['giocatori'] = df['giocatori'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        if 'vincitori' in df.columns and not df['vincitori'].empty:
            df['vincitori'] = df['vincitori'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        
        for col in ['num_giocatori', 'punti_vittoria', 'punti_bonus']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
        else:
            df['data'] = pd.NaT

        for col in COLONNE_LOG:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[COLONNE_LOG]
        return df.astype({'punti_bonus': 'float', 'punti_vittoria': 'float', 'num_giocatori': 'int'})
    except Exception as e:
        st.warning(f"Errore nel caricamento del log: {e}. Provo a creare un log vuoto.")
        return pd.DataFrame(columns=COLONNE_LOG)

def salva_log_gsheet(df):
    """Salva l'intero DataFrame sul Foglio Google, sovrascrivendolo."""
    if 'gs_worksheet' in st.session_state and st.session_state.gs_worksheet is not None:
        try:
            df_to_save = df.copy()
            df_to_save['giocatori'] = df_to_save['giocatori'].astype(str)
            df_to_save['vincitori'] = df_to_save['vincitori'].astype(str)
            
            st.session_state.gs_worksheet.clear()
            set_with_dataframe(st.session_state.gs_worksheet, df_to_save, include_index=False, resize=True)
            return True
        except Exception as e:
            st.sidebar.error(f"Errore nel salvataggio su GSheets: {e}")
            return False
    return False

# --- 3. Funzioni di Logica del Torneo ---

def inizializza_stato():
    """Inizializza lo stato in modo robusto."""
    
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if 'log_partite' not in st.session_state:
        st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
    
    if 'classifica' not in st.session_state:
        classifica_vuota = pd.DataFrame(
            0.0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:]
        )
        classifica_vuota['PG'] = 0
        classifica_vuota['Elo'] = ELO_STARTING
        st.session_state.classifica = classifica_vuota.reset_index().rename(columns={'index': 'Giocatore'})
    
    if 'gs_worksheet' not in st.session_state:
        st.session_state.gs_worksheet = None
        
    if 'log_caricato' not in st.session_state:
        st.session_state.log_caricato = False

    if st.session_state.gs_worksheet is None:
        st.session_state.gs_worksheet = connect_to_gsheet()
    
    if st.session_state.gs_worksheet is not None and not st.session_state.log_caricato:
        st.session_state.log_partite = carica_log_gsheet(st.session_state.gs_worksheet)
        st.session_state.log_caricato = True
        ricalcola_classifica()
        
def reset_torneo():
    """Resetta il torneo CANCELLANDO i dati sul Foglio Google."""
    st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
    salva_log_gsheet(st.session_state.log_partite)
    
    classifica_vuota = pd.DataFrame({
        'Giocatore': LISTA_GIOCATORI,
        'PG': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'PT': 0.0, 'MPP': 0.0,
        'Elo': ELO_STARTING
    }).set_index('Giocatore')
    st.session_state.classifica = classifica_vuota.reset_index()
    
    st.session_state["password_correct"] = False
    st.sidebar.success("Torneo resettato.")

def check_password():
    """Restituisce True se la password è corretta, False altrimenti."""
    if st.session_state.get("password_correct", False):
        return True
    st.title("🔒 Accesso Protetto")
    st.write("Inserisci la password per accedere al torneo:")
    try:
        correct_password = st.secrets["credentials"]["password"]
    except:
        st.error("Password non configurata. Controlla il file .streamlit/secrets.toml.")
        return False
    password_input = st.text_input("Password", type="password", key="password_input_widget")
    if st.button("Accedi"):
        if password_input == correct_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Password errata. Riprova.")
    return False

def ricalcola_classifica():
    """Ricalcola l'intera classifica (MPP e ELO) basandosi sul log."""
    log = st.session_state.get('log_partite', pd.DataFrame(columns=COLONNE_LOG))
    
    classifica_nuova = pd.DataFrame(
        0.0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:]
    )
    classifica_nuova['PG'] = 0
    classifica_nuova['Elo'] = ELO_STARTING
    
    if log.empty or log.dropna(subset=['data']).empty:
        st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})
        return

    # --- 1. Calcolo Sistema MPP (Aggregato) ---
    log_valido = log.dropna(subset=['giocatori', 'vincitori', 'data'])
    
    pg_counts = log_valido['giocatori'].explode().value_counts()
    classifica_nuova['PG'].update(pg_counts)
    
    for partita in log_valido.itertuples():
        col_vittoria = f'V{int(partita.num_giocatori)}'
        if col_vittoria in classifica_nuova.columns:
            for vincitore in partita.vincitori:
                if vincitore in classifica_nuova.index:
                    classifica_nuova.loc[vincitore, col_vittoria] += 1
                    classifica_nuova.loc[vincitore, 'PT'] += (partita.punti_vittoria + partita.punti_bonus)
    
    classifica_nuova['MPP'] = np.where(
        classifica_nuova['PG'] > 0, 
        classifica_nuova['PT'] / classifica_nuova['PG'], 
        0
    ).round(3) 

    # --- 2. Calcolo Sistema ELO (Sequenziale) ---
    log_ordinato_elo = log_valido.sort_values(by="data", ascending=True)
    elo_correnti = classifica_nuova['Elo'].to_dict()
    
    for partita in log_ordinato_elo.itertuples():
        giocatori = partita.giocatori
        vincitori = partita.vincitori
        perdenti = [p for p in giocatori if p not in vincitori]
        num_g = partita.num_giocatori

        try:
            if num_g == 2: # 1v1
                p_vinc = vincitori[0]
                p_perd = perdenti[0]
                R_vinc, R_perd = elo_correnti[p_vinc], elo_correnti[p_perd]
                E_vinc = 1 / (1 + 10**((R_perd - R_vinc) / 400))
                delta = ELO_K_FACTOR * (1 - E_vinc)
                elo_correnti[p_vinc] += delta
                elo_correnti[p_perd] -= delta

            elif num_g == 4: # 2v2
                R_team_vinc = (elo_correnti[vincitori[0]] + elo_correnti[vincitori[1]]) / 2
                R_team_perd = (elo_correnti[perdenti[0]] + elo_correnti[perdenti[1]]) / 2
                E_team_vinc = 1 / (1 + 10**((R_team_perd - R_team_vinc) / 400))
                delta_team = ELO_K_FACTOR * (1 - E_team_vinc)
                elo_correnti[vincitori[0]] += delta_team
                elo_correnti[vincitori[1]] += delta_team
                elo_correnti[perdenti[0]] -= delta_team
                elo_correnti[perdenti[1]] -= delta_team

            elif num_g == 3: # 1v1v1
                p_vinc = vincitori[0]
                p_perd1, p_perd2 = perdenti[0], perdenti[1]
                R_vinc = elo_correnti[p_vinc]
                R_perd_avg = (elo_correnti[p_perd1] + elo_correnti[p_perd2]) / 2
                E_vinc = 1 / (1 + 10**((R_perd_avg - R_vinc) / 400))
                delta_totale = ELO_K_FACTOR * (1 - E_vinc)
                elo_correnti[p_vinc] += delta_totale
                elo_correnti[p_perd1] -= delta_totale / 2
                elo_correnti[p_perd2] -= delta_totale / 2
        
        except (KeyError, IndexError) as e:
             st.warning(f"Errore nel processare una partita per Elo (giocatore non trovato o dati errati). Partita saltata.")
             continue
    
    classifica_nuova['Elo'] = classifica_nuova.index.map(elo_correnti).round(0).astype(int)
    st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})


def registra_partita(giocatori_partita, vincitori_selezionati, bonus_attivo):
    """Aggiunge una partita al log, lo SALVA SU GSHEET e ricalcola."""
    
    num_giocatori = len(giocatori_partita)
    num_vincitori = len(vincitori_selezionati)
    if not giocatori_partita: st.sidebar.error("Seleziona i giocatori."); return False
    if num_giocatori < 2: st.sidebar.error("Seleziona almeno 2 giocatori."); return False
    if not vincitori_selezionati: st.sidebar.error("Seleziona almeno un vincitore."); return False
    if num_giocatori not in PUNTI_MAP: st.sidebar.error(f"Il numero di giocatori ({num_giocatori}) non è valido."); return False
    if num_giocatori in [2, 3] and num_vincitori != 1: st.sidebar.error(f"Le partite a {num_giocatori} giocatori devono avere 1 solo vincitore."); return False
    if num_giocatori == 4 and num_vincitori != 2: st.sidebar.error("Le partite a 4 giocatori devono avere 2 vincitori."); return False
    for v in vincitori_selezionati:
        if v not in giocatori_partita: st.sidebar.error(f"Il vincitore {v} non è tra i giocatori."); return False

    data_ora = datetime.now()
    giorno = data_ora.strftime('%A')
    punti_base = PUNTI_MAP[num_giocatori]
    punti_bonus_val = PUNTI_BONUS_100 if bonus_attivo else 0.0
    
    nuova_partita = pd.DataFrame([{
        "data": data_ora, "giorno_settimana": giorno, "giocatori": giocatori_partita, 
        "vincitori": vincitori_selezionati, "num_giocatori": num_giocatori,
        "punti_vittoria": punti_base, "punti_bonus": punti_bonus_val
    }])
    
    st.session_state.log_partite = pd.concat(
        [st.session_state.log_partite, nuova_partita], 
        ignore_index=True
    )
    
    salva_log_gsheet(st.session_state.log_partite)
    ricalcola_classifica()
    
    vincitori_str = " e ".join(vincitori_selezionati)
    messaggio_bonus = f" (+{PUNTI_BONUS_100} bonus)" if bonus_attivo else ""
    st.sidebar.success(f"Partita registrata! {vincitori_str} vincono {punti_base}{messaggio_bonus} punti (MPP).")
    st.toast("Classifiche aggiornate!", icon="🏆")
    
    return True

# --- 4. Struttura dell'App Streamlit (MODIFICATA) ---

def main():
    # --- MODIFICA UI: Aggiunto icona ---
    st.set_page_config(page_title="Torneo Briscola", layout="wide", page_icon="🏆")
    
    inizializza_stato()

    if not check_password():
        st.stop()
    
    def processa_registrazione():
        giocatori = st.session_state.multiselect_giocatori
        vincitori = st.session_state.select_vincitori
        bonus = st.session_state.check_bonus 
        successo = registra_partita(giocatori, vincitori, bonus)
        if successo:
            st.session_state.multiselect_giocatori = []
            st.session_state.select_vincitori = []
            st.session_state.check_bonus = False

    # --- Sidebar (invariata) ---
    with st.sidebar:
        st.header("📋 Registra Partita")
        
        giocatori_selezionati = st.multiselect(
            "Chi ha giocato?", options=LISTA_GIOCATORI, key="multiselect_giocatori"
        )
        vincitori_selezionati = st.multiselect( 
            "Chi ha vinto?", options=giocatori_selezionati if giocatori_selezionati else [], key="select_vincitori" 
        )
        
        if vincitori_selezionati:
            st.checkbox(f"Bonus >100 punti (+{PUNTI_BONUS_100}pt)", key="check_bonus")
        else:
            st.checkbox(f"Bonus >100 punti (+{PUNTI_BONUS_100}pt)", key="check_bonus", disabled=True, value=False)

        st.button(
            "Registra Partita", use_container_width=True, type="primary", on_click=processa_registrazione
        )
        st.divider()
        if st.button("🗑️ Elimina Ultima Partita", use_container_width=True):
            if st.session_state.log_partite.empty:
                st.sidebar.error("Nessuna partita da eliminare.")
            else:
                st.session_state.log_partite = st.session_state.log_partite.sort_values(by="data").iloc[:-1]
                salva_log_gsheet(st.session_state.log_partite)
                ricalcola_classifica()
                st.sidebar.success("Ultima partita eliminata con successo.")
                st.rerun()

        if st.button("🚨 RESETTA TORNEO 🚨", use_container_width=True):
            reset_torneo()
            st.rerun()

    # --- PAGINA PRINCIPALE CON NUOVA UI ---
    st.title("🏆 Torneo di Briscola")

    tab_mpp, tab_elo = st.tabs(["📊 Classifica Torneo (MPP)", "👑 Classifica Rating (Elo)"])
    
    # Prepara la classifica base (serve per i podi)
    classifica_base = st.session_state.classifica.copy()
    if 'PG' in classifica_base.columns and not classifica_base.empty:
        max_pg = int(classifica_base['PG'].max())
    else:
        max_pg = 0

    # --- Tab 1: Classifica MPP (REVAMPED) ---
    with tab_mpp:
        st.header("📊 Classifica Torneo (MPP)")
        st.markdown("Ordinata per **MPP (Media Punti per Partita)**. Premia l'efficienza.")

        # Calcola podio non filtrato
        classifica_mpp_podio = classifica_base.sort_values(
            by=["MPP", "PT", "PG"], ascending=[False, False, True]
        ).reset_index(drop=True)

        # --- MODIFICA UI: PODIO (Top 3) ---
        col1, col2, col3 = st.columns(3)
        if len(classifica_mpp_podio) >= 1:
            r = classifica_mpp_podio.iloc[0]
            col1.metric("🥇 1° Posto", r['Giocatore'], f"{r['MPP']:.3f} MPP ({r['PG']} PG)")
        if len(classifica_mpp_podio) >= 2:
            r = classifica_mpp_podio.iloc[1]
            col2.metric("🥈 2° Posto", r['Giocatore'], f"{r['MPP']:.3f} MPP ({r['PG']} PG)")
        if len(classifica_mpp_podio) >= 3:
            r = classifica_mpp_podio.iloc[2]
            col3.metric("🥉 3° Posto", r['Giocatore'], f"{r['MPP']:.3f} MPP ({r['PG']} PG)")
        st.divider()

        # --- MODIFICA LOGICA: Default value=2 ---
        soglia_pg = st.slider(
            "Mostra solo giocatori con almeno X Partite Giocate (PG):", 
            min_value=0, max_value=max_pg + 1, value=2, # <-- DEFAULT 2
            help="Usare 0 per mostrare tutti, default 2 (più di 1 partita).",
            key="slider_mpp"
        )
        
        # Filtra e mostra la tabella
        classifica_mpp_filtrata = classifica_base.copy()
        if soglia_pg > 0:
            classifica_mpp_filtrata = classifica_mpp_filtrata[classifica_mpp_filtrata['PG'] >= soglia_pg]
        
        st.dataframe(
            classifica_mpp_filtrata.sort_values(by=["MPP", "PT", "PG"], ascending=[False, False, True]), 
            use_container_width=True,
            column_config={ "PT": st.column_config.NumberColumn(format="%.1f"), "MPP": st.column_config.NumberColumn(format="%.3f"), "Elo": st.column_config.NumberColumn(format="%d") }
        )

    # --- Tab 2: Classifica Elo (REVAMPED) ---
    with tab_elo:
        st.header("👑 Classifica Rating (Elo)")
        st.markdown(f"Ordinata per **Rating Elo**. Misura la forza relativa (Start: {ELO_STARTING}, K-Factor: {ELO_K_FACTOR}).")

        # Calcola podio non filtrato
        classifica_elo_podio = classifica_base.sort_values(
            by=["Elo", "MPP", "PG"], ascending=[False, False, True]
        ).reset_index(drop=True)

        # --- MODIFICA UI: PODIO (Top 3) ---
        col1_e, col2_e, col3_e = st.columns(3)
        if len(classifica_elo_podio) >= 1:
            r = classifica_elo_podio.iloc[0]
            col1_e.metric("🥇 1° Posto", r['Giocatore'], f"{r['Elo']} Elo ({r['PG']} PG)")
        if len(classifica_elo_podio) >= 2:
            r = classifica_elo_podio.iloc[1]
            col2_e.metric("🥈 2° Posto", r['Giocatore'], f"{r['Elo']} Elo ({r['PG']} PG)")
        if len(classifica_elo_podio) >= 3:
            r = classifica_elo_podio.iloc[2]
            col3_e.metric("🥉 3° Posto", r['Giocatore'], f"{r['Elo']} Elo ({r['PG']} PG)")
        st.divider()
        
        # --- MODIFICA LOGICA: Default value=2 ---
        soglia_pg_elo = st.slider(
            "Mostra solo giocatori con almeno X Partite Giocate (PG):", 
            min_value=0, max_value=max_pg + 1, value=2, # <-- DEFAULT 2
            help="Usare 0 per mostrare tutti, default 2 (più di 1 partita).",
            key="slider_elo"
        )

        # Filtra e mostra la tabella
        classifica_elo_filtrata = classifica_base.copy()
        if soglia_pg_elo > 0:
            classifica_elo_filtrata = classifica_elo_filtrata[classifica_elo_filtrata['PG'] >= soglia_pg_elo]

        st.dataframe(
            classifica_elo_filtrata.sort_values(by=["Elo", "MPP", "PG"], ascending=[False, False, True]), 
            use_container_width=True,
            column_config={ "Elo": st.column_config.NumberColumn(format="%d"), "PT": st.column_config.NumberColumn(format="%.1f"), "MPP": st.column_config.NumberColumn(format="%.3f") }
        )

    # --- MODIFICA UI: Log partite nascosto in un Expander ---
    with st.expander("Mostra/Nascondi Log Partite Giocate", expanded=False):
        st.header("📜 Log Partite Giocate")
        if st.session_state.log_partite.empty:
            st.info("Nessuna partita ancora registrata.")
        else:
            st.dataframe(
                st.session_state.log_partite.sort_values(by="data", ascending=False),
                use_container_width=True,
                column_config={
                    "data": st.column_config.DatetimeColumn("Data e Ora", format="DD/MM/YYYY - HH:mm"),
                    "giocatori": st.column_config.ListColumn("Giocatori Coinvolti"),
                    "vincitori": st.column_config.ListColumn("Vincitori"),
                    "punti_bonus": st.column_config.NumberColumn("Bonus", format="%.1f")
                }
            )

if __name__ == "__main__":
    main()