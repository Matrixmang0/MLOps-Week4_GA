from datetime import timedelta
from feast import BigQuerySource, FeatureView, FeatureService, Entity, ValueType

# Define flower species as entity
species_id = Entity(
    name="species_id",
    description="Species of Iris Flower",
    value_type=ValueType.INT64
)

# Define feature view for flower measurements
iris_features = FeatureView(
    name="iris_features",
    entities=[species_id],
    ttl=timedelta(days=1),
    source=BigQuerySource(
        table="neural-mantra-461520-m0.Offline_Store.Iris",
        timestamp_field="event_timestamp"
    ),
)

# Create feature service for one model version
# FeatureService groups features for specific use cases
model = FeatureService(
    name="iris_model",
    features=[iris_features]
)