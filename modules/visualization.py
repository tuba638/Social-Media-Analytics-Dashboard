def prepare_chart_data(sentiments):
    labels = list(sentiments.keys())
    values = list(sentiments.values())
    return labels, values