import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers                                                                                                                                                                                                         # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint                                                                                                                                                                       # type: ignore

def main() -> None: # Train a 4-class CNN and save the trained model
    # Resolve project paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    
    train_dir = project_root / "app" / "data" / "brain_mri" / "Training"
    test_dir = project_root / "app" / "data" / "brain_mri" / "Testing"
    model_dir = project_root / "app" / "data" / "saved_models"
    model_dir.mkdir(parents = True, exist_ok = True)
    
    # final model used by Flask
    model_path = model_dir / "brain_tumor_cnn_multiclass.h5"
    # best checkpoint during training
    checkpoint_path = model_dir / "brain_tumor_cnn_multiclass_best.h5"
    
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Training dir: {train_dir}")
    print(f"[INFO] Testing  dir: {test_dir}")
    print(f"[INFO] Model will be saved to: {model_path}")
    
    if not train_dir.exists():
        raise FileNotFoundError(
           f"Training directory not found at {train_dir}\n"
            "Expected structure:\n"
            "  Training/glioma\n"
            "  Training/meningioma\n"
            "  Training/pituitary\n"
            "  Training/no_tumor\n"
        )
    
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Testing directory not found at {test_dir}\n"
            "Expected structure:\n"
            "  Testing/glioma\n"
            "  Testing/meningioma\n"
            "  Testing/pituitary\n"
            "  Testing/no_tumor\n"
        )
        
    # Dataset configuration
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    SEED = 42

    # training & validation datasets
    print("[INFO] Creating training and validation datasets...")
    
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels = "inferred",        # folder name ('yes'/'no') -> label
        label_mode = "int",      # # multi-class labels: 0, 1, 2, 3
        validation_split = VALIDATION_SPLIT,
        subset = "training",
        seed = SEED,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        color_mode = "rgb",
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels = "inferred",
        label_mode = "int",
        validation_split = VALIDATION_SPLIT,
        subset = "validation",
        seed = SEED,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        color_mode = "rgb",
    )
    
    # Test dataset from Testing
    print("[INFO] Creating test dataset...")
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels = "inferred",
        label_mode = "int",
        seed = SEED,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        color_mode = "rgb",
        shuffle = False,
    )
    
    class_names = train_ds.class_names  # ['no', 'yes'] -> label 0 = 'no', label 1 = 'yes'
    num_classes = len(class_names)
    print(f"[INFO] Class names (from folders): {class_names}")
    print(f"[INFO] Number of classes: {num_classes}")
    
    # Prepare datasets -> cache / prefetch / normalization / augmentation
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Prefetch for performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Normalization layer: scales [0, 255] -> [0, 1]
    normalization_layer = layers.Rescaling(1.0 / 255)
    
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ],
        name = "data_augmentation",
    )
    
    # Build CNN model
    print("[INFO] Building CNN model...")

    inputs = keras.Input(shape = (*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    
    x = layers.Conv2D(32, (3, 3), activation = "relu", padding = "same")(x) # create diffrent filters, One might look for vertical lines, another for horizontal lines, another for curves
    x = layers.MaxPooling2D((2, 2))(x) # reduces image size by half to reduce amount of math calc needed for next layer
    
    x = layers.Conv2D(64, (3, 3), activation = "relu", padding = "same")(x)
    x = layers.MaxPooling2D((2, 2))(x) 

    x = layers.Conv2D(128, (3, 3), activation = "relu", padding = "same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation = "relu", padding = "same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x) # final decision-making layers (Dense) understand flat lists, not 3D images
    x = layers.Dense(256, activation = "relu")(x)   # thinking happens
    x = layers.Dropout(0.5)(x) # solve overfitting problem by forcing model to learn robust patterns
    
    # Output: single neuron, sigmoid -> probability of class "1" ('yes')
    outputs = layers.Dense(num_classes, activation = "softmax")(x)
    model = keras.Model(inputs, outputs, name="brain_tumor_cnn_multiclass")  # Wraps everything up into a single object
    model.summary(print_fn = lambda line: print("[MODEL] " + line))
    
    # Compile the model
    model.compile(
        optimizer = "adam", # Adaptive Moment Estimation
        loss = "sparse_categorical_crossentropy", # calculates how wrong the model is
        metrics = ["accuracy"], 
    )
    
    early_stop = EarlyStopping(
    monitor = 'val_loss',    # Watch the validation loss
    patience = 7,            # Wait 7 epochs before stopping
    restore_best_weights = True, # Go back to the "sweet spot" version
    verbose = 1,
    )
    
    checkpoint_cb = ModelCheckpoint(
        filepath = str(checkpoint_path),
        monitor = "val_loss",
        save_best_only = True,
        verbose = 1,
    )
    
    # Train the model
    EPOCHS = 100
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS,
        callbacks = [early_stop, checkpoint_cb]
    )
    
    if checkpoint_path.exists():
        print(f"[INFO] Loading best weights from checkpoint: {checkpoint_path}")
        model.load_weights(str(checkpoint_path))
    
    # Evaluate on validation data
    print("[INFO] Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"[RESULT] Validation loss: {val_loss:.4f}")
    print(f"[RESULT] Validation acc:  {val_acc:.4f}")

    print("[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[RESULT] Test loss: {test_loss:.4f}")
    print(f"[RESULT] Test acc:  {test_acc:.4f}")
    
    # Save the trained model
    model.save(model_path)
    print(f"[INFO] Multi-class model saved to: {model_path}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()