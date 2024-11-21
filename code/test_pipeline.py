from main import Pipeline

def test_basic_pipeline():
    """Basic test to ensure pipeline runs"""
    pipeline = Pipeline('config.yaml')  # Use local config file
    results = pipeline.run(['hup', 'mni'])
    assert results is not None

if __name__ == "__main__":
    test_basic_pipeline()