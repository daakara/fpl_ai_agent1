# Team Recommender

This project is designed to assist users in building optimal fantasy football teams by providing recommendations based on player data, fixture difficulty, and various scoring metrics. The system employs iterative optimization techniques to refine team selections and maximize expected points while adhering to budget constraints.

## Project Structure

- **data/**
  - `players.csv`: Contains player data including attributes such as player ID, name, position, cost, expected points, and injury status.
  - `fixtures.csv`: Contains fixture data that includes match dates, teams involved, and difficulty ratings for upcoming matches.

- **src/**
  - `team_recommender.py`: Main logic for team recommendations, including functions for generating team selections and implementing iterative optimization.
  - `data_loader.py`: Responsible for loading and preprocessing data from CSV files into pandas DataFrames.
  - `evaluator.py`: Functions for evaluating team performance based on selected metrics, such as expected points and team diversity.
  - `strategy.py`: Implements various strategies for team selection, including bench optimization and captaincy selection.
  - `utils.py`: Utility functions that support the main logic, such as calculating team chemistry bonuses.

- **notebooks/**
  - `data_exploration.ipynb`: Jupyter notebook for exploratory data analysis on player and fixture data.
  - `model_testing.ipynb`: Jupyter notebook for testing different models and strategies for team recommendations.

- **tests/**
  - `test_team_recommender.py`: Unit tests for the functions in `team_recommender.py`.
  - `test_utils.py`: Unit tests for utility functions defined in `utils.py`.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd team_recommender
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

To generate team recommendations, you can use the `get_latest_team_recommendations` function from the `team_recommender.py` module. This function allows you to specify various parameters such as budget, formations, and player selection constraints.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.