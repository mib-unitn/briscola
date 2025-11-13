import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np

# --- 1. Configurazione Iniziale ---

# !!! MODIFICA QUI I TUOI GIOCATORI !!!
LISTA_GIOCATORI = ["Anna", "Bruno", "Carla", "Davide", "Elena", "Franco"]

# Mappa dei punti base
PUNTI_MAP = {
    2: 3,  # Vittoria in partita a 2
    3: 5,  # Vittoria in partita a 3
    4: 2   # Vittoria in partita a 4 (per ciascun vincitore)
}
# Valore del bonus
PUNTI_BONUS_100 = 0.5

# Colonne per i nostri DataFrame
COLONNE_LOG = ["data", "giorno_settimana", "giocatori", "vincitori", "num_giocatori", "punti_vittoria", "punti_bonus"]
COLONNE_CLASSIFICA = ["Giocatore", "PG", "V2", "V3", "V4", "PT", "MPP"]

# --- 2. Funzioni di Logica del Torneo ---

def inizializza_stato():
    """Inizializza i DataFrame nello stato della sessione se non esistono."""
    
    if 'log_partite' not in st.session_state:
        st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
    
    if 'classifica' not in st.session_state:
        classifica_vuota = pd.DataFrame({
            'Giocatore': LISTA_GIOCATORI,
            'PG': 0, 'V2': 0, 'V3': 0, 'V4': 0,
            'PT': 0.0,
            'MPP': 0.0
        }).set_index('Giocatore')
        st.session_state.classifica = classifica_vuota.reset_index()
        
    # --- NUOVO: Inizializza lo stato della password ---
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        
# --- NUOVA FUNZIONE: Blocco Password ---
def check_password():
    """
    Restituisce True se la password è corretta, False altrimenti.
    Mostra il form di login.
    """
    
    # Se la password è già stata data ed è corretta, non fare nulla
    if st.session_state.get("password_correct", False):
        return True

    # Mostra il form di login
    st.title("🔒 Accesso Protetto")
    st.write("Inserisci la password per accedere al torneo:")

    # Carica la password corretta da secrets.toml
    try:
        correct_password = st.secrets["credentials"]["password"]
    except:
        st.error("Password non configurata. Crea il file .streamlit/secrets.toml come da istruzioni.")
        return False
        
    password_input = st.text_input("Password", type="password", key="password_input_widget")
    
    if st.button("Accedi"):
        if password_input == correct_password:
            # Password corretta! Salva nello stato e ricarica la pagina
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Password errata. Riprova.")
            
    return False # La password non è (ancora) corretta

def ricalcola_classifica():
    """
    Ricalcola l'intera classifica da zero basandosi sul log_partite.
    Gestisce correttamente i vincitori multipli e i punti bonus.
    """
    log = st.session_state.log_partite
    
    classifica_nuova = pd.DataFrame(0, index=LISTA_GIOCATORI, columns=COLONNE_CLASSIFICA[1:-1]) 
    classifica_nuova['PT'] = classifica_nuova['PT'].astype(float) 
    classifica_nuova['MPP'] = 0.0

    if log.empty:
        st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})
        return

    # 1. Calcola Partite Giocate (PG)
    pg_counts = log['giocatori'].explode().value_counts()
    classifica_nuova['PG'].update(pg_counts)
    
    # 2. Calcola Vittorie (V2, V3, V4) e Punti Totali (PT)
    for partita in log.itertuples():
        vincitori_list = partita.vincitori
        num_g = partita.num_giocatori
        
        punti_base = partita.punti_vittoria
        punti_bonus = partita.punti_bonus
        punti_totali_partita = punti_base + punti_bonus
        
        col_vittoria = f'V{num_g}'
        
        if col_vittoria in classifica_nuova.columns:
            for vincitore in vincitori_list:
                if vincitore in classifica_nuova.index:
                    classifica_nuova.loc[vincitore, col_vittoria] += 1
                    classifica_nuova.loc[vincitore, 'PT'] += punti_totali_partita
    
    # 3. Calcola Media Punti per Partita (MPP)
    classifica_nuova['MPP'] = np.where(
        classifica_nuova['PG'] > 0, 
        classifica_nuova['PT'] / classifica_nuova['PG'], 
        0
    ).round(3) 
    
    st.session_state.classifica = classifica_nuova.reset_index().rename(columns={'index': 'Giocatore'})


def registra_partita(giocatori_partita, vincitori_selezionati, bonus_attivo):
    """
    Aggiunge una nuova partita al log e ricalcola la classifica.
    Accetta 'bonus_attivo' dalla checkbox.
    Restituisce True se la registrazione ha successo, False altrimenti.
    """
    
    # 1. Validazione
    num_giocatori = len(giocatori_partita)
    num_vincitori = len(vincitori_selezionati)

    if not giocatori_partita:
        st.sidebar.error("Seleziona i giocatori.")
        return False
    if num_giocatori < 2:
        st.sidebar.error("Seleziona almeno 2 giocatori.")
        return False
    if not vincitori_selezionati:
        st.sidebar.error("Seleziona almeno un vincitore.")
        return False
    if num_giocatori not in PUNTI_MAP:
        st.sidebar.error(f"Il numero di giocatori ({num_giocatori}) non è valido. Ammessi: 2, 3, o 4.")
        return False
    if num_giocatori in [2, 3] and num_vincitori != 1:
        st.sidebar.error(f"Le partite a {num_giocatori} giocatori devono avere 1 solo vincitore.")
        return False
    if num_giocatori == 4 and num_vincitori != 2:
        st.sidebar.error("Le partite a 4 giocatori devono avere 2 vincitori.")
        return False
    for v in vincitori_selezionati:
        if v not in giocatori_partita:
            st.sidebar.error(f"Il vincitore {v} non è tra i giocatori selezionati.")
            return False

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
    
    # 4. Aggiorna il log
    st.session_state.log_partite = pd.concat(
        [st.session_state.log_partite, nuova_partita], 
        ignore_index=True
    )
    
    # 5. Ricalcola tutto
    ricalcola_classifica()
    
    vincitori_str = " e ".join(vincitori_selezionati)
    messaggio_bonus = f" (+{PUNTI_BONUS_100} bonus)" if bonus_attivo else ""
    st.sidebar.success(f"Partita registrata! {vincitori_str} vincono {punti_base}{messaggio_bonus} punti ciascuno.")
    st.toast("Classifica aggiornata!", icon="🏆")
    
    return True

# --- 3. Struttura dell'App Streamlit ---

def main():
    st.set_page_config(page_title="Torneo Briscola", layout="wide")
    
    # Inizializza stato PER PRIMO, così "password_correct" esiste
    inizializza_stato()

    # --- BLOCCO PASSWORD ---
    # Controlla la password. Se non è corretta, ferma l'app qui.
    if not check_password():
        st.stop()
    # --- FINE BLOCCO ---
    
    # --- Se la password è corretta, il resto dell'app viene disegnato ---

    # --- Funzione di Callback per il Bottone ---
    def processa_registrazione():
        giocatori = st.session_state.multiselect_giocatori
        vincitori = st.session_state.select_vincitori
        bonus = st.session_state.check_bonus 
        
        successo = registra_partita(giocatori, vincitori, bonus)
        
        if successo:
            st.session_state.multiselect_giocatori = []
            st.session_state.select_vincitori = []
            st.session_state.check_bonus = False

    # --- Colonna Laterale (Sidebar) per Input ---
    with st.sidebar:
        st.header("📋 Registra Partita")
        
        giocatori_selezionati = st.multiselect(
            "Chi ha giocato?",
            options=LISTA_GIOCATORI,
            key="multiselect_giocatori"
        )
        
        vincitori_selezionati = st.multiselect( 
            "Chi ha vinto?",
            options=giocatori_selezionati if giocatori_selezionati else [],
            key="select_vincitori" 
        )
        
        if vincitori_selezionati:
            st.checkbox(f"Bonus >100 punti (+{PUNTI_BONUS_100}pt)", key="check_bonus")
        else:
            st.checkbox(f"Bonus >100 punti (+{PUNTI_BONUS_100}pt)", key="check_bonus", disabled=True, value=False)

        st.button(
            "Registra Partita", 
            use_container_width=True, 
            type="primary",
            on_click=processa_registrazione
        )
            
        st.divider()
        if st.button("🚨 RESETTA TORNEO 🚨", use_container_width=True):
            st.session_state.log_partite = pd.DataFrame(columns=COLONNE_LOG)
            inizializza_stato() # Resetta tutto, inclusa la password
            st.rerun()

    # --- Pagina Principale (Main) per Visualizzazione ---
    st.title("🏆 Classifica Torneo di Briscola")
    st.markdown("La classifica è ordinata per **MPP (Media Punti per Partita)** per bilanciare il numero di partite giocate.")

    max_pg = int(st.session_state.classifica['PG'].max())
    soglia_pg = st.slider(
        "Mostra solo giocatori con almeno X Partite Giocate (PG):", 
        min_value=0, 
        max_value=max_pg + 1,
        value=0,
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
