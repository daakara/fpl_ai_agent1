# FPL Analytics Pro

## Overview
FPL Analytics Pro is an advanced Fantasy Premier League (FPL) analytics application designed to provide users with insights and recommendations based on player performance, fixture difficulty ratings, and AI-powered suggestions. The application leverages data analysis and visualization techniques to enhance the FPL experience.

## Features
- **Player Analysis**: Analyze player performance metrics and identify top performers.
- **Fixture Difficulty Ratings (FDR)**: Evaluate the difficulty of upcoming fixtures based on attack and defense for the next 5 games.
- **AI Recommendations**: Get smart player recommendations based on advanced metrics and analysis.
- **Team Builder**: Build and optimize your FPL team using enhanced analytics.
- **Visualizations**: Create insightful visualizations to understand player and team performance.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fpl_ai_agent
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
streamlit run fpl_ai_agent/simple_app.py
```

Once the application is running, you can navigate through the various tabs to explore player analysis, fixture difficulty ratings, and more.

## Data Sources
The application uses JSON files for fixture schedules and team strength data, which are essential for calculating fixture difficulty ratings. Ensure that the data files are correctly placed in the `fpl_ai_agent/data/fixtures/` directory.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please create a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.