import requests
def get_price_change_candidates():
    url = "https://api.fplstatistics.co.uk/pricechanges"
    try:
        resp = requests.get(url)
        data = resp.json()
        # Example: Find players due to rise tonight
        to_rise = [p for p in data if p["target"] >= 100]
        return to_rise
    except Exception as e:
        return []
