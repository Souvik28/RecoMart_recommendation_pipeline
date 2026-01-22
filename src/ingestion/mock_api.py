from flask import Flask, jsonify
import random

app = Flask(__name__)

# Mock product metadata for cold start problem
PRODUCT_METADATA = {
    f"P{i}": {
        "product_id": f"P{i}",
        "category": random.choice(["Electronics", "Books", "Clothing", "Home", "Sports"]),
        "price": round(random.uniform(10, 500), 2),
        "brand": random.choice(["BrandA", "BrandB", "BrandC", "BrandD"]),
        "avg_rating": round(random.uniform(2.0, 5.0), 1),
        "popularity_score": random.randint(1, 100)
    } for i in range(101, 151)
}

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify(list(PRODUCT_METADATA.values()))

@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    return jsonify(PRODUCT_METADATA.get(product_id, {}))

if __name__ == '__main__':
    print("Starting Mock Product API on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)