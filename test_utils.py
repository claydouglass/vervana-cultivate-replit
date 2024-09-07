from utils import (
    predict_yield, 
    analyze_plant_health_trends, 
    predict_pest_disease_risk, 
    optimize_nutrient_plan, 
    fetch_and_store_historical_data,
    get_adjustment_recommendations,
    import_recipes,
    compare_batch_performance,
    early_warning_system,
    batch_exists,
    import_cultivation_schedule,  # Add this import
    import_processing_schedule,  # Add this import
    schedule_cultivation_tasks,  # Add this import
    schedule_processing_tasks,  # Add this import
    get_cultivation_schedule,  # Add this import
    get_processing_schedule  # Add this import
)
from models import db, EnvironmentalData, BatchData, NutrientData, CultivationSchedule, ProcessingSchedule  # Add CultivationSchedule and ProcessingSchedule
from main import app
from datetime import datetime, timedelta

def test_functions():
    with app.app_context():
        print("Testing fetch_and_store_historical_data:")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Test with 7 days of data
        print(fetch_and_store_historical_data(start_date, end_date))
        
        print("\nTesting import_recipes:")
        recipes = import_recipes()
        print(f"Imported {len(recipes)} recipes")
        print("First recipe:", recipes[0])
        
        # Fetch or create TEST001 batch
        batch = BatchData.query.filter_by(batch_id="TEST001").first()
        if not batch:
            batch = BatchData(
                batch_number=1,
                batch_id="TEST001",
                veg_week_1_2_start=datetime.now() - timedelta(days=60),
                veg_week_1_2_end=datetime.now() - timedelta(days=46),
                veg_week_3_start=datetime.now() - timedelta(days=45),
                veg_week_3_end=datetime.now() - timedelta(days=39),
                flower_week_1_3_start=datetime.now() - timedelta(days=38),
                flower_week_1_3_end=datetime.now() - timedelta(days=18),
                flower_week_4_6_5_start=datetime.now() - timedelta(days=17),
                flower_week_4_6_5_end=datetime.now() - timedelta(days=3),
                flower_week_6_5_8_5_start=datetime.now() - timedelta(days=2),
                flower_week_6_5_8_5_end=datetime.now(),
                harvest_date=datetime.now(),
                yield_amount=500,
                thc_level=20,
                terpene_profile="myrcene:0.5,limonene:0.3,pinene:0.2"
            )
            db.session.add(batch)
            db.session.commit()

        print("\nTesting optimize_nutrient_plan:")
        print(optimize_nutrient_plan("TEST001"))
        
        print("\nTesting compare_batch_performance:")
        # Fetch or create TEST002 batch
        batch2 = BatchData.query.filter_by(batch_id="TEST002").first()
        if not batch2:
            batch2 = BatchData(
                batch_number=2,
                batch_id="TEST002",
                veg_week_1_2_start=datetime.now() - timedelta(days=90),
                veg_week_1_2_end=datetime.now() - timedelta(days=76),
                veg_week_3_start=datetime.now() - timedelta(days=75),
                veg_week_3_end=datetime.now() - timedelta(days=69),
                flower_week_1_3_start=datetime.now() - timedelta(days=68),
                flower_week_1_3_end=datetime.now() - timedelta(days=48),
                flower_week_4_6_5_start=datetime.now() - timedelta(days=47),
                flower_week_4_6_5_end=datetime.now() - timedelta(days=33),
                flower_week_6_5_8_5_start=datetime.now() - timedelta(days=32),
                flower_week_6_5_8_5_end=datetime.now() - timedelta(days=18),
                flower_week_8_5_plus_start=datetime.now() - timedelta(days=17),
                flower_week_8_5_plus_end=datetime.now() - timedelta(days=1),
                harvest_date=datetime.now() - timedelta(days=1),
                yield_amount=450,
                thc_level=18,
                terpene_profile="myrcene:0.4,limonene:0.4,pinene:0.1"
            )
            db.session.add(batch2)
            db.session.commit()
        
        # Add some test environmental and nutrient data for both batches
        for i in range(30):
            env_data1 = EnvironmentalData(
                timestamp=batch.veg_week_1_2_start + timedelta(days=i),
                temperature=25 + i * 0.1,
                humidity=50 + i * 0.2,
                co2_level=1000 + i * 5,
                vpd=1.0 + i * 0.05,
                light_duration=16 if i % 2 == 0 else 8,
                is_day=i % 2 == 0
            )
            env_data2 = EnvironmentalData(
                timestamp=batch2.veg_week_1_2_start + timedelta(days=i),
                temperature=24 + i * 0.1,
                humidity=52 + i * 0.2,
                co2_level=980 + i * 5,
                vpd=0.9 + i * 0.05,
                light_duration=16 if i % 2 == 0 else 8,
                is_day=i % 2 == 0
            )
            nutrient_data1 = NutrientData(
                timestamp=batch.veg_week_1_2_start + timedelta(days=i),
                nitrogen_level=150 + i,
                phosphorus_level=50 + i * 0.5,
                potassium_level=200 + i * 0.8
            )
            nutrient_data2 = NutrientData(
                timestamp=batch2.veg_week_1_2_start + timedelta(days=i),
                nitrogen_level=145 + i,
                phosphorus_level=48 + i * 0.5,
                potassium_level=205 + i * 0.8
            )
            db.session.add_all([env_data1, env_data2, nutrient_data1, nutrient_data2])
        
        db.session.commit()
        
        comparison_result = compare_batch_performance("TEST001", "TEST002")
        print("Batch Comparison Result:")
        print(f"Number of periodic comparisons: {len(comparison_result['stage_comparisons'])}")
        if comparison_result['stage_comparisons']:
            print("First periodic comparison:", comparison_result['stage_comparisons'][0])
        print("Final comparison:", comparison_result['final_comparison'])
        
        print("\nTesting early_warning_system:")
        warnings = early_warning_system()
        print("Early Warning System Results:")
        print(warnings)
        
        print("\nTesting import_cultivation_schedule:")
        cultivation_schedule = import_cultivation_schedule()
        print(f"Imported {len(cultivation_schedule)} days of cultivation schedule")
        print("First day cultivation schedule:", cultivation_schedule[0])

        print("\nTesting import_processing_schedule:")
        processing_schedule = import_processing_schedule()
        print(f"Imported {len(processing_schedule)} days of processing schedule")
        print("First day processing schedule:", processing_schedule[0])

        print("\nTesting schedule_cultivation_tasks:")
        cultivation_result = schedule_cultivation_tasks("TEST001", 1)  # Schedule tasks for day 1
        print(cultivation_result)

        print("\nTesting schedule_processing_tasks:")
        processing_result = schedule_processing_tasks("TEST001", 1)  # Schedule tasks for day 1
        print(processing_result)

        print("\nTesting get_cultivation_schedule:")
        cultivation_schedule = get_cultivation_schedule("TEST001", 1)  # Get schedule for day 1
        print(cultivation_schedule)

        print("\nTesting get_processing_schedule:")
        processing_schedule = get_processing_schedule("TEST001", 1)  # Get schedule for day 1
        print(processing_schedule)

def test_detect_new_batch_start():
    with app.app_context():
        # Create test data
        room_id = "TEST_ROOM"
        for i in range(48):
            env_data = EnvironmentalData(
                timestamp=datetime.now() - timedelta(hours=48-i),
                room_id=room_id,
                light_duration=18 if i < 24 else 15,  # Transition at 24 hours ago
                temperature=25,
                humidity=50,
                co2_level=1000,
                light_intensity=500,
                vpd=1.0,
                is_day=True
            )
            db.session.add(env_data)
        db.session.commit()

        # Test the function
        new_batch_start = detect_new_batch_start(room_id)
        assert new_batch_start is not None, "Failed to detect new batch start"
        print(f"New batch start detected at: {new_batch_start}")

        # Clean up test data
        EnvironmentalData.query.filter_by(room_id=room_id).delete()
        db.session.commit()

# Add this to your test_functions()
test_detect_new_batch_start()

if __name__ == "__main__":
    test_functions()