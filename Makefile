clean:
	rm config.json
	rm meta.json
	rm dq.json
	cp vanilla/* .
	rm encoder.pkl
	rm data/batches/*
	rm summary/*
	rm models/MLPRegressor/*
	rm models/PARegressor/*
	rm models/SGDRegressor/*
	rm models/best_model.pkl