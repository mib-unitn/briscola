import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import gspread # <-- NUOVO
from gspread_dataframe import get_as_dataframe, set_with_dataframe # <-- NUOVO
import ast

# --- 1. Configurazione Iniziale ---

# !!! INCOLLA QUI L'URL DEL TUO FOGLIO GOOGLE !!!
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ea2DT-FlYmq_TjA4MFqdc1rr3GFj1MxpVfWm0MQxrQo/edit?gid=0#gid=0" 
# !!! ASSICURATI DI AVERLO CONDIVISO CON L'EMAIL DEL BOT !!!


# !!! MODIFICA QUI I TUOI GIOCATORI !!!
LISTA_GIOCATORI = ["Michele", "Federico", "Lorenza", "Grazia", "Pierpaolo", "Niccolò", "Simone", "Avinash", "Sahil", "Massine", "Beppe", "Esteban"]

# Mappa dei punti base
PUNTI_MAP = {
    2: 3,  # Vittoria in partita a 2
    3: 5,  # Vittoria in partita a 3
    4: 2   # Vittoria in partita a 4 (per ciascun vincitore)
}
PUNTI_BONUS_100 = 0.5

# Colonne
COLONNE_LOG = ["data", "giorno_settimana", "giocatori", "vincitori", "num_giocatori", "punti_vittoria", "punti_bonus"]
COLONNE_CLASSIFICA = ["Giocatore", "PG", "V2", "V3", "V4", "PT", "MPP"]

# --- 2. Funzioni di Persistenza Google Sheets (NUOVE) ---

def connect_to_gsheet():
    """Connettiti a Google Sheets usando i secrets di Streamlit."""
    try:
        # Usa i secrets per l'autenticazione
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        
        # Apri il foglio usando il suo URL
        worksheet = gc.open_by_url(GOOGLE_SHEET_URL).sheet1
        return worksheet
    except Exception as e:
        st.error(f"Errore di connessione a Google Sheets: {e}")
        st.error("Assicurati di aver condiviso il foglio con l'email 'client_email' del bot.")
        return None

def carica_log_gsheet(worksheet):
    """Carica il log dal Foglio Google."""
    try:
        # Ottieni tutti i dati e caricali in un DataFrame
        df = get_as_dataframe(worksheet, evaluate_formulas=True, dtype_backend='pyarrow')

        # Se il foglio è vuoto, df.columns sarà vuoto. Inizializziamo.
        if df.empty:
            df = pd.DataFrame(columns=COLONNE_LOG)
        
        # Pulisci eventuali righe/colonne vuote
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Ricostruisci le colonne se non esistono (primo avvio)
        if df.empty:
             df = pd.DataFrame(columns=COLONNE_LOG)
             
        # Converti le colonne da stringhe a liste (Google Sheets salva le liste come stringhe)
        if 'giocatori' in df.columns and not df['giocatori'].empty:
            df['giocatori'] = df['giocatori'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        if 'vincitori' in df.columns and not df['vincitori'].empty:
            df['vincitori'] = df['vincitori'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.startswith('[') else x)
        
        # Assicura che le colonne numeriche siano numeriche
        for col in ['num_giocatori', 'punti_vittoria', 'punti_bonus']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Assicura che le colonne del log esistano
        for col in COLONNE_LOG:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[COLONNE_LOG] # Riordina e seleziona solo le colonne giuste
        
        return df.astype({'punti_bonus': 'float', 'punti_vittoria': 'float', 'num_giocatori': 'int'})

    except Exception as e:
        st.warning(f"Errore nel caricamento del log: {e}. Provo a creare un log vuoto.")
        # Se c'è un errore (es. foglio vuoto o corrotto), crea un log vuoto
        return pd.DataFrame(columns=COLONNE_LOG)

def salva_log_gsheet(df):
    """Salva l'intero DataFrame sul Foglio Google, sovrascrivendolo."""
    if 'gs_worksheet' in st.session_state and st.session_state.gs_worksheet is not None:
        try:
            # Converte le liste in stringhe per salvarle
            df_to_save = df.copy()
            df_to_save['giocatori'] = df_to_save['giocatori'].astype(str)
            df_to_save['vincitori'] = df_to_save['vincitori'].astype(str)
            
            # Pulisce il foglio e scrive il nuovo DataFrame
            st.session_state.gs_worksheet.clear()
            set_with_dataframe(st.session_state.gs_worksheet, df_to_save, include_index=False, resize=True)
            return True
        except Exception as e:
            st.sidebar.error(f"Errore nel salvataggio su GSheets: {e}")
            return False
    return False

# --- 3. Funzioni di Logica del Torneo (MODIFICATE) ---

def inizializza_stato():
    """
    Inizializza lo stato: si connette a GSheets e carica il log *una sola volta*.
    """
    
    # Connessione a Google Sheets
    if 'gs_worksheet' not in st.session_state:
        st.session_state.gs_worksheet = connect_to_gsheet()

    # Caricamento dati
    if 'log_caricato' not in st.session_state and st.session_state.gs_worksheet is not None:
        st.session_state.log_partite = carica_log_gsheet(st.session_state.gs_worksheet)
        st.session_state.log_caricato = True
        
        if 'classifica' not in st.session_state:
            classifica_vuota = pd.DataFrame(
                0.0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:]
            )
            classifica_vuota['PG'] = 0
            st.session_state.classifica = classifica_vuota.reset_index().rename(columns={'index': 'Giocatore'})
        
        ricalcola_classifica()

    # Gestione password
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        
def reset_torneo():
    """Resetta il torneo CANCELLANDO i dati sul Foglio Google."""
    
    st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
    
    # Salva il log vuoto su Google Sheets
    salva_log_gsheet(st.session_state.log_partite)
    
    classifica_vuota = pd.DataFrame({
        'Giocatore': LISTA_GIOCATORI,
        'PG': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'PT': 0.0, 'MPP': 0.0
    }).set_index('Giocatore')
    st.session_state.classifica = classifica_vuota.reset_index()
    
    st.session_state["password_correct"] = False
    st.sidebar.success("Torneo resettato.")

# ... (Funzione check_password() rimane IDENTICA a prima) ...
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

# ... (Funzione ricalcola_classifica() rimane IDENTICA a prima) ...
def ricalcola_classifica():
    """Ricalcola l'intera classifica basandosi sul log in st.session_state."""
    log = st.session_state.get('log_partite', pd.DataFrame(columns=COLONNE_LOG))
    
    classifica_nuova = pd.DataFrame(0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:-1]) 
    classifica_nuova['PT'] = classifica_nuova['PT'].astype(float) 
    classifica_nuova['MPP'] = 0.0

    if log.empty:
        st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})
        return

    # Aggiungi filtro per log con dati non validi
    log_valido = log.dropna(subset=['giocatori', 'vincitori'])

    pg_counts = log_valido['giocatori'].explode().value_counts()
    classifica_nuova['PG'].update(pg_counts)
    
    for partita in log_valido.itertuples():
        vincitori_list = partita.vincitori
        num_g = partita.num_giocatori
        punti_base = partita.punti_vittoria
        punti_bonus = partita.punti_bonus
        punti_totali_partita = punti_base + punti_bonus
        col_vittoria = f'V{int(num_g)}' # Assicura che num_g sia intero
        
        if col_vittoria in classifica_nuova.columns:
            for vincitore in vincitori_list:
                if vincitore in classifica_nuova.index:
                    classifica_nuova.loc[vincitore, col_vittoria] += 1
                    classifica_nuova.loc[vincitore, 'PT'] += punti_totali_partita
    
    classifica_nuova['MPP'] = np.where(
        classifica_nuova['PG'] > 0, 
        classifica_nuova['PT'] / classifica_nuova['PG'], 
        0
    ).round(3) 
    
    st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})


def registra_partita(giocatori_partita, vincitori_selezionati, bonus_attivo):
    """Aggiunge una partita al log, lo SALVA SU GSHEET e ricalcola."""
    
    # 1. Validazione (identica a prima)
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

    # 2. Raccogli dati
    data_ora = datetime.now()
    giorno = data_ora.strftime('%A')
    punti_base = PUNTI_MAP[num_giocatori]
    punti_bonus_val = PUNTI_BONUS_100 if bonus_attivo else 0.0
    
    # 3. Crea la nuova riga per il log
    nuova_partita = pd.DataFrame([{
        "data": data_ora,
        "giorno_settimana": giorno,
        "giocatori": giocatori_partita, 
        "vincitori": vincitori_selezionati,
        "num_giocatori": num_giocatori,
        "punti_vittoria": punti_base,
        "punti_bonus": punti_bonus_val
    }])
    
    # 4. Aggiorna il log in memoria
    st.session_state.log_partite = pd.concat(
        [st.session_state.log_partite, nuova_partita], 
        ignore_index=True
    )
    
    # 5. --- SALVA SU GOOGLE SHEETS --- (MODIFICA CHIAVE)
    salva_log_gsheet(st.session_state.log_partite)
    
    # 6. Ricalcola classifica
    ricalcola_classifica()
    
    vincitori_str = " e ".join(vincitori_selezionati)
    messaggio_bonus = f" (+{PUNTI_BONUS_100} bonus)" if bonus_attivo else ""
    st.sidebar.success(f"Partita registrata! {vincitori_str} vincono {punti_base}{messaggio_bonus} punti ciascuno.")
    st.toast("Classifica aggiornata!", icon="🏆")
    
    return True

# --- 4. Struttura dell'App Streamlit ---

def main():
    st.set_page_config(page_title="Torneo Briscola", layout="wide")
    
    # Carica i dati da GSheet in session_state all'avvio
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
                st.session_state.log_partite = st.session_state.log_partite.iloc[:-1]
                
                # --- SALVA SU GOOGLE SHEETS --- (MODIFICA CHIAVE)
                salva_log_gsheet(st.session_state.log_partite)
                
                ricalcola_classifica()
                st.sidebar.success("Ultima partita eliminata con successo.")
                st.rerun()

        if st.button("🚨 RESETTA TORNEO 🚨", use_container_width=True):
            reset_torneo()
            st.rerun()

    # --- Pagina Principale (Main) per Visualizzazione ---
    st.title("🏆 Classifica Torneo di Briscola")
    st.markdown("La classifica è ordinata per **MPP (Media Punti per Partita)** per bilanciare il numero di partite giocate.")

    if 'classifica' in st.session_state and 'PG' in st.session_state.classifica.columns and not st.session_state.classifica.empty:
        max_pg = int(st.session_state.classifica['PG'].max())
    else:
        max_pg = 0
        
    soglia_pg = st.slider(
        "Mostra solo giocatori con almeno X Partite Giocate (PG):", 
        min_value=0, max_value=max_pg + 1, value=0,
        help="Usare 0 per mostrare tutti i giocatori."
    )
    
    classifica_da_mostrare = st.session_state.classifica.copy()
    
    if soglia_pg > 0:
        classifica_da_mostrare = classifica_da_mostrare[classifica_da_mostrare['PG'] >= soglia_pg]

    classifica_ordinata = classifica_da_mostrare.sort_values(
        by=["MPP", "PT", "PG"], 
        ascending=[False, False, True]
    ).reset_index(drop=True)
    
    st.dataframe(
        classifica_ordinata, 
        use_container_width=True,
        column_config={
            "PT": st.column_config.NumberColumn("PT (Punti Totali)", format="%.1f"),
            "MPP": st.column_config.NumberColumn("MPP", format="%.3f"),
        }
    )

    st.divider()
    st.header("📜 Log Partite Giocate")
    
    if st.session_state.log_partite.empty:
        st.info("Nessuna partita ancora registrata.")
    else:
        st.dataframe(
            st.session_state.log_partite.sort_values(by="data", ascending=False).drop(columns=['data']), # Rimuoviamo la colonna data che non serve più
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
