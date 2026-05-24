Prerekviziti:
    WSL2 potreban za Windows mašine.
    uv pip install -r requirements_streamlit.txt

Pokretanje:
    source .venv_linux/bin/activate

    python -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType none

    StreamLit konstantno gleda promene u fajlovima, pa je potrebno isključiti tu opciju da ne bi došlo do problema sa performansama (Pošto hostujem linux u windows-u).


Folderi:
demo_output - ostatak od SIR 2
mobility_lab - kod za aplikaciju
mobiML - skinuta biblioteka sa gita
sumo_output - simulacije


Napomene:

Vi ste zatražili od mene da pokrenem opet SUMO, ne kroz Web Wizard već kroz terminal, takođe ste pomenuli da probam da loopujem trajektorije automobila (tj. da ne izađu iz simulacije), medjutim nisam našao tu komandu, pa sam umesto toga dodavao nove automobile tokom vremena, tako da se simulacija ne isprazni na sredini timeline-a.

Takođe sam hteo da proširim simulaciju ka stopshopu, ali nisam uspeo da pokrenem download ponovo.
U suštini je problem u osmGet.py, gde pokušavam da pribavim OpenStreetMap za novu zonu koju sam izabrao preko lat/lon, ali dobijam grešku 429 (Too Many Requests), iako nisam mnogo puta query-ovao server, uspelo mi je samo jedanput, i to kada sam pokušavao da pokrenem to za SIR 2 još pre više od mesec dana.

Aplikacija trenutno koristi my_simulations\2026-05-22-11-15-56 kao base, što je novija simulacija sa više vozila, ali sa istom zonom Niša.

