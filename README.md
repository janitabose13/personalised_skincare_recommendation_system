# Personalized Skincare Recommendation System

## Overview
This project is an AI-driven personalized skincare recommendation system designed to analyze skincare products, ingredients, and user preferences to generate tailored skincare recommendations.

The system leverages datasets from platforms like Nykaa, INCIDecoder, and EWG to provide insights into product suitability based on ingredients and skin concerns.


## Objectives
- Build a recommendation system for skincare products
- Analyze ingredients for safety and effectiveness
- Provide personalized suggestions based on user needs
- Enable data-driven skincare decisions



## Project Structure
```markdown
personalised_skincare_recommendation_system/
│── data/
│ ├── raw/ # Original datasets (Nykaa, EWG, INCIDecoder)
│ ├── processed/ # Cleaned and transformed data
│
│── Code/ # Source code for preprocessing & modeling
│── requirements.txt # Dependencies
│── README.md # Project documentation
```



## Datasets Used
- Nykaa product and review dataset
- INCIDecoder ingredient dataset
- EWG ingredient safety dataset



## Installation

1. Clone the repository:
```bash
git clone https://github.com/janitabose13/personalised_skincare_recommendation_system.git
cd personalised_skincare_recommendation_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
- Load and preprocess datasets from /data/raw
- Clean and transform data into /data/processed
- Run recommendation models from the /Code folder


## Features
- Ingredient analysis
- Product recommendation engine
- Dataset integration from multiple sources
- Scalable ML pipeline


## Tech Stack
- Python
- Pandas, NumPy
- Machine Learning (Scikit-learn / others)
- Data Processing Pipelines



## Future Improvements
- Add deep learning models
- Build a user interface (web/app)
- Integrate real-time user input
- Improve recommendation accuracy


## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.



## License
This project is for academic and research purposes.



## Author
Janita Bose


