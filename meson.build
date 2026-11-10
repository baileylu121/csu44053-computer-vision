project(
    'pool-balls',
    'cpp',
    default_options: ['cpp_std=c++26', 'warning_level=3'],
)

opencv = dependency('opencv4')

executable(
    'app',
    'main.cpp',
    'myApplication.cpp',
    dependencies: opencv,
    install: true,
)

# Local tool for generating distorted test images; not part of the submission.
executable('augment', 'tools/augment.cpp', dependencies: opencv)
executable('evaluate', 'tools/evaluate.cpp', dependencies: opencv)
executable('stages', 'tools/stages.cpp', dependencies: opencv)
