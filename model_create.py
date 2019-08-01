import turicreate as tc

images = tc.load_images('./images/')
images['labels'] = images['path'].element_slice(9,-4)

model = tc.one_shot_object_detector.create(images, 'labels')

predictions = model.predict(data)

# Export to Core ML
model.export_coreml('grn.mlmodel')
