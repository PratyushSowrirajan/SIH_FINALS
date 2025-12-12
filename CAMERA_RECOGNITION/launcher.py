"""
Advanced Face Recognition System - Main Launcher
Provides menu to train faces or run real-time recognition
"""

import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║      ADVANCED FACE MESH RECOGNITION SYSTEM               ║
║                                                          ║
║      High-Accuracy Real-Time Face Recognition            ║
║      with Dense Mesh Visualization                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

def main_menu():
    while True:
        clear_screen()
        print_banner()
        
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("\n1. Train New Face (Capture 200 samples)")
        print("2. Run Face Recognition (Real-time)")
        print("3. View Trained Models")
        print("4. Expression Detection (Original)")
        print("5. Exit")
        print("\n" + "="*60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            clear_screen()
            print("\n" + "="*60)
            print("FACE TRAINING MODE")
            print("="*60)
            
            from face_trainer import FaceMeshTrainer
            trainer = FaceMeshTrainer()
            
            user_name = input("\nEnter name for training: ").strip()
            if user_name:
                num_frames = 200
                response = input(f"Capture {num_frames} frames? (Y/n): ").strip().lower()
                
                if response and response != 'y':
                    try:
                        num_frames = int(input("Enter number of frames: "))
                    except:
                        print("Invalid input. Using 200 frames.")
                        num_frames = 200
                
                trainer.capture_training_samples(user_name, num_frames)
            
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            clear_screen()
            print("\n" + "="*60)
            print("FACE RECOGNITION MODE")
            print("="*60)
            
            from face_recognizer import AdvancedFaceRecognizer
            recognizer = AdvancedFaceRecognizer()
            recognizer.run()
        
        elif choice == '3':
            clear_screen()
            print("\n" + "="*60)
            print("TRAINED MODELS")
            print("="*60)
            
            import pickle
            model_dir = 'trained_models'
            
            if not os.path.exists(model_dir):
                print("\nNo models found.")
            else:
                models = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
                
                if not models:
                    print("\nNo trained models found.")
                else:
                    print(f"\nFound {len(models)} trained model(s):\n")
                    
                    for i, filename in enumerate(models, 1):
                        try:
                            with open(os.path.join(model_dir, filename), 'rb') as f:
                                model = pickle.load(f)
                                print(f"{i}. {model['user_name']}")
                                print(f"   - Samples: {model['num_samples']}")
                                print(f"   - Trained: {model['trained_date']}")
                                print(f"   - Threshold: {model['recognition_threshold']}")
                                print()
                        except Exception as e:
                            print(f"{i}. {filename} (Error loading: {e})")
            
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            clear_screen()
            print("\n" + "="*60)
            print("EXPRESSION DETECTION MODE")
            print("="*60)
            print("\nLaunching original expression detection system...\n")
            
            import main
            main.main()
        
        elif choice == '5':
            clear_screen()
            print("\n" + "="*60)
            print("Thank you for using the Face Recognition System!")
            print("="*60 + "\n")
            break
        
        else:
            print("\n✗ Invalid option. Please select 1-5.")
            input("Press Enter to continue...")


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        input("\nPress Enter to exit...")
