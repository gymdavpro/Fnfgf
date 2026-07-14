[app]
# (str) Version of the application
version = 1.0

# (str) Package name
package.name = gfgandataset

# (str) Package domain (needed for android packaging)
package.domain = org.david

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas

# (list) Application requirements
# Обязательно добавляем numpy и pillow для работы нейросети и датасета
requirements = python3, kivy, numpy, pillow

# (str) Supported orientations (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (list) Permissions
# Запрашиваем доступ к чтению и записи во внутреннюю память телефона для загрузки датасета
android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, MANAGE_EXTERNAL_STORAGE

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK will support.
android.minapi = 21

# (bool) Use Demangled names in android manifest
android.ndk_api = 21

# (str) Android NDK directory (if empty, it will be automatically downloaded)
# android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded)
# android.sdk_path =

# (list) Android GPU architectures to build for
android.archs = arm64-v8a

# (bool) Allow backup
android.allow_backup = True

# (str) The format used to package the app for release mode (aab or apk)
android.release_artifact = apk

# (str) The format used to package the app for debug mode (apk or aab)
android.debug_artifact = apk

[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1
