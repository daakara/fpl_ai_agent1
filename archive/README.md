# README.md
# FPL AI Agent

An advanced Fantasy Premier League analysis and team recommendation tool.

## 📁 Project Structure

```
fpl_ai_agent/
├── core/                    # Main application files
│   ├── simple_app.py       # Primary Streamlit app
│   └── main_app.py         # Streamlined entry point
├── components/              # Reusable components
│   ├── team_recommender.py # Team optimization
│   ├── fpl_official.py     # FPL API integration
│   ├── data_loader.py      # Data fetching
│   ├── llm_integration.py  # AI chat features
│   └── ui_components.py    # UI components
├── config/                  # Configuration
│   └── config_template.py  # App configuration
├── utils/                   # Utility functions
│   └── utils.py            # Helper functions
├── data/                    # Data files
├── archive/                 # Archived/unused files
│   ├── legacy_apps/        # Old app versions
│   ├── test_files/         # Test scripts
│   ├── unused_components/  # Unused modules
│   └── old_configs/        # Old config files
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd fpl_ai_agent
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp config/.env.template .env
   # Edit .env with your API keys
   ```

3. **Run the app:**
   ```bash
   streamlit run core/simple_app.py
   ```

## 🔧 Configuration

Set these environment variables in `.env`:
- `OPENAI_API_KEY` - For AI chat features
- `COHERE_API_KEY` - Alternative AI provider
- `FPL_TEAM_ID` - Your FPL team ID (optional)

## 📦 Features

- 📊 **Live FPL Data** - Real-time player stats and fixtures
- 🤖 **AI Chat** - Ask questions about FPL strategy
- 🏆 **Team Recommendations** - Optimized team suggestions
- 📈 **Advanced Analytics** - Player performance insights
- 💎 **Value Picks** - Find hidden gems and differentials

## 🗂️ Archive

The `archive/` folder contains:
- Old app versions that are no longer maintained
- Test files and debugging scripts
- Unused components that may be needed later
- Old configuration files

## 🤝 Contributing

1. Keep new features in `components/`
2. Update tests in `archive/test_files/` if needed
3. Follow the existing code structure
4. Update this README for major changes