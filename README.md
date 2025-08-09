# Football AI Predictions 🎯⚽

Intelligente Fußballvorhersagen mit Machine Learning für die Top 5 europäischen Ligen.

## 🚀 Features

- **Automatische Vorhersagen**: Zweimal wöchentlich (Dienstag 14:00, Freitag 18:00)
- **Machine Learning**: Ensemble-Modell mit Random Forest und Gradient Boosting
- **Confidence Scoring**: High Confidence Vorhersagen mit >65% Genauigkeit
- **Live-Statistiken**: ROI-Tracking und Performance-Metriken
- **Responsive Design**: Modernes UI für alle Geräte

## 📊 Unterstützte Ligen

- 🇩🇪 Bundesliga (D1)
- 🏴󐁧󐁢󐁥󐁮󐁧󐁿 Premier League (E0)
- 🇫🇷 Ligue 1 (F1)
- 🇮🇹 Serie A (I1)
- 🇪🇸 La Liga (SP1)

## 🛠️ Installation

### Lokale Installation

1. Repository klonen:
```bash
git clone https://github.com/yourusername/football-prediction-app.git
cd football-prediction-app
```

2. Virtuelle Umgebung erstellen:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

3. Dependencies installieren:
```bash
pip install -r requirements.txt
```

4. Setup ausführen:
```bash
python setup_data.py
```

5. App starten:
```bash
python app.py
```

Die App läuft nun unter `http://localhost:5000`

### Railway Deployment

1. Repository zu GitHub pushen
2. Neues Projekt in Railway erstellen
3. GitHub Repository verbinden
4. Deployment startet automatisch

## 📁 Projektstruktur

```
football-prediction-app/
├── app.py                 # Haupt-Flask-Anwendung
├── prediction_agent.py    # Automatisierter Vorhersage-Runner
├── models/               
│   ├── predictor.py      # ML-Vorhersagemodell
│   └── comparator.py     # Ergebnisvergleich
├── templates/            # HTML-Templates
├── static/               # CSS/JS Assets
├── data/                 # Spieldaten
├── predictions/          # Vorhersagen
└── results/              # Ergebnisse
```

## 🔧 Konfiguration

Die Konfiguration erfolgt über `config.json`:

```json
{
  "data": {
    "current_season": "2025_26",
    "leagues": ["D1"],
    "update_schedule": {
      "predictions": ["Tuesday 14:00", "Friday 18:00"],
      "results": ["Daily 23:00"]
    }
  }
}
```

## 📈 API Endpoints

- `GET /` - Dashboard mit Statistiken
- `GET /predictions` - Aktuelle Vorhersagen
- `GET /results` - Vergangene Ergebnisse
- `GET /api/stats` - JSON-Statistiken
- `GET /api/predictions/latest` - Neueste Vorhersagen (JSON)

## 🧪 Testing

```bash
# Unit Tests
python -m pytest tests/

# Mit Coverage
python -m pytest --cov=. tests/
```

## 🤝 Contributing

1. Fork das Repository
2. Feature Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add some AmazingFeature'`)
4. Branch pushen (`git push origin feature/AmazingFeature`)
5. Pull Request öffnen

## ⚠️ Disclaimer

Diese App dient ausschließlich Bildungszwecken. Keine Vorhersage ist garantiert. 
Bitte wetten Sie verantwortungsvoll und nur mit Geld, dessen Verlust Sie sich leisten können.

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

## 📧 Kontakt

Bei Fragen oder Problemen:
- Email: your-email@example.com
- GitHub Issues: [Issues](https://github.com/yourusername/football-prediction-app/issues)

## 🙏 Danksagungen

- [Football-Data.co.uk](https://www.football-data.co.uk/) für die Datenbereitstellung
- Railway für das Hosting
- Scikit-learn für ML-Tools