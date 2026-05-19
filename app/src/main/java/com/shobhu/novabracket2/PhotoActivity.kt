package com.shobhu.novabracket2

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class PhotoActivity : AppCompatActivity() {

    companion object {
        private const val TAG              = "PhotoDetector"
        private const val PICK_IMAGE_REQ   = 101
        private const val CAMERA_REQ       = 102
        private const val CAMERA_PERM_REQ  = 201
        private const val MODEL_FILE       = "best_float16.tflite"
        private const val LABELS_FILE      = "classes.txt"
        private const val MODEL_INPUT_SIZE = 640
        private const val NUM_CHANNELS = 11   // matches your model [1, 11, 8400]
        private const val NUM_PREDS    = 8400
        private const val NUM_CLASSES  = NUM_CHANNELS - 4   // = 7 classes
        private const val NMS_IOU          = 0.45f
        private const val THRESHOLD_DEF    = 0.25f
        private const val THRESHOLD_MIN    = 0.01f
        private const val THRESHOLD_MAX    = 0.99f
    }

    private lateinit var imageView:      ImageView
    private lateinit var overlayView:    PhotoOverlayView
    private lateinit var btnPick:        Button
    private lateinit var btnCamera:      Button
    private lateinit var btnRerun:       Button
    private lateinit var tvStatus:       TextView
    private lateinit var seekBar:        SeekBar
    private lateinit var tvThreshold:    TextView
    private lateinit var editThreshold:  EditText
    private lateinit var btnApply:       Button
    private lateinit var progressBar:    ProgressBar
    private lateinit var imageContainer: FrameLayout

    private var photoUri: Uri? = null   // URI of the temp file the camera writes into

    private var interpreter: Interpreter? = null
    private var labels: List<String>      = emptyList()
    private var confThreshold             = THRESHOLD_DEF

    private lateinit var inferenceThread: HandlerThread
    private lateinit var inferenceHandler: Handler

    private var currentBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )

        inferenceThread  = HandlerThread("InferenceThread").also { it.start() }
        inferenceHandler = Handler(inferenceThread.looper)

        loadModel()
        loadLabels()
        buildUI()
        window.statusBarColor = Color.BLACK  // must be after setContentView (called inside buildUI)
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceThread.quitSafely()
        interpreter?.close()
    }

    // ─────────────────────────────────────────
    //  UI
    // ─────────────────────────────────────────
    private fun buildUI() {
        val dp = resources.displayMetrics.density

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#050A0F"))
        }

        // ── Top bar ───────────────────────────────────────────────────────
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER_VERTICAL
            setBackgroundColor(Color.parseColor("#0A1414"))
            setPadding(
                (12 * dp).toInt(), (14 * dp).toInt(),
                (16 * dp).toInt(), (14 * dp).toInt()
            )
        }

        val btnBack = TextView(this).apply {
            text     = "‹  BACK"
            textSize = 14f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.parseColor("#00FFCC"))
            setPadding((8 * dp).toInt(), (8 * dp).toInt(), (16 * dp).toInt(), (8 * dp).toInt())
        }

        val topTitle = TextView(this).apply {
            text     = "PHOTO DETECTION"
            textSize = 15f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.WHITE)
            letterSpacing = 0.1f
            layoutParams  = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            gravity       = Gravity.CENTER_HORIZONTAL
        }

        topBar.addView(btnBack)
        topBar.addView(topTitle)
        // Spacer to balance the back button width
        topBar.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams((60 * dp).toInt(), 1)
        })

        root.addView(topBar, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ))

        btnBack.setOnClickListener { finish() }

        // ── Image area ────────────────────────────────────────────────────
        imageContainer = FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#08121A"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }

        imageView = ImageView(this).apply {
            scaleType   = ImageView.ScaleType.FIT_CENTER
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        overlayView = PhotoOverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        // Placeholder when no image selected
        val placeholder = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity     = Gravity.CENTER
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }
        val phIcon = TextView(this).apply {
            text = "🖼"; textSize = 64f; gravity = Gravity.CENTER
        }
        val phText = TextView(this).apply {
            text = "Tap 'UPLOAD PHOTO' to begin"
            textSize = 13f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
            setTextColor(Color.parseColor("#2A4A4A"))
            gravity = Gravity.CENTER
        }
        placeholder.addView(phIcon)
        placeholder.addView(phText)
        placeholder.tag = "placeholder"

        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
            layoutParams = FrameLayout.LayoutParams(
                (56 * dp).toInt(), (56 * dp).toInt(), Gravity.CENTER
            )
            indeterminateTintList =
                android.content.res.ColorStateList.valueOf(Color.parseColor("#00FFCC"))
        }

        imageContainer.addView(placeholder)
        imageContainer.addView(imageView)
        imageContainer.addView(overlayView)
        imageContainer.addView(progressBar)
        root.addView(imageContainer)

        // ── Status text ───────────────────────────────────────────────────
        tvStatus = TextView(this).apply {
            text     = "No image loaded"
            textSize = 11f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
            setTextColor(Color.parseColor("#607070"))
            gravity  = Gravity.CENTER
            setPadding(0, (6 * dp).toInt(), 0, (6 * dp).toInt())
        }
        root.addView(tvStatus, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ))

        // ── Threshold row ─────────────────────────────────────────────────
        val threshPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0A1414"))
            setPadding(
                (16 * dp).toInt(), (10 * dp).toInt(),
                (16 * dp).toInt(), (10 * dp).toInt()
            )
        }

        val threshRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER_VERTICAL
        }

        val threshLabel = TextView(this).apply {
            text     = "THRESHOLD"
            textSize = 11f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.parseColor("#607070"))
            letterSpacing = 0.1f
            layoutParams  = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { marginEnd = (10 * dp).toInt() }
        }

        seekBar = SeekBar(this).apply {
            max      = 98
            progress = ((confThreshold - THRESHOLD_MIN) * 100).toInt()
            progressTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#00FFCC"))
            thumbTintList    = android.content.res.ColorStateList.valueOf(Color.parseColor("#00FFCC"))
            layoutParams     = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            ).apply { marginEnd = (10 * dp).toInt() }
        }

        tvThreshold = TextView(this).apply {
            text     = "%.2f".format(confThreshold)
            textSize = 14f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.parseColor("#00FFCC"))
            layoutParams = LinearLayout.LayoutParams(
                (44 * dp).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT
            )
            gravity = Gravity.END or Gravity.CENTER_VERTICAL
        }

        threshRow.addView(threshLabel)
        threshRow.addView(seekBar)
        threshRow.addView(tvThreshold)

        // Manual input row
        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (8 * dp).toInt() }
        }

        editThreshold = EditText(this).apply {
            setText("%.2f".format(confThreshold))
            hint      = "0.01 – 0.99"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                    android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            textSize  = 13f
            setTextColor(Color.WHITE)
            setHintTextColor(Color.parseColor("#445555"))
            setBackgroundColor(Color.parseColor("#111D1D"))
            setPadding(
                (10 * dp).toInt(), (7 * dp).toInt(),
                (10 * dp).toInt(), (7 * dp).toInt()
            )
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_DONE
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            ).apply { marginEnd = (8 * dp).toInt() }
        }

        btnApply = Button(this).apply {
            text = "Apply"
            textSize = 12f
            setTextColor(Color.BLACK)
            setBackgroundColor(Color.parseColor("#00FFCC"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        inputRow.addView(editThreshold)
        inputRow.addView(btnApply)

        threshPanel.addView(threshRow)
        threshPanel.addView(inputRow)
        root.addView(threshPanel, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ))

        // ── Bottom buttons ────────────────────────────────────────────────
        val btnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER
            setPadding(
                (16 * dp).toInt(), (14 * dp).toInt(),
                (16 * dp).toInt(), (20 * dp).toInt()
            )
            setBackgroundColor(Color.parseColor("#050A0F"))
        }

        // Camera button — opens device camera
        btnCamera = Button(this).apply {
            text     = "📷  CAMERA"
            textSize = 13f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.parseColor("#050A0F"))
            setBackgroundColor(Color.parseColor("#00FFCC"))
            layoutParams = LinearLayout.LayoutParams(0,
                (52 * dp).toInt(), 1f
            ).apply { marginEnd = (8 * dp).toInt() }
        }

        btnPick = Button(this).apply {
            text     = "📁  GALLERY"
            textSize = 13f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.parseColor("#FF6B35"))
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(Color.TRANSPARENT)
                setStroke((2 * dp).toInt(), Color.parseColor("#FF6B35"))
                cornerRadius = 4 * dp
            }
            layoutParams = LinearLayout.LayoutParams(0,
                (52 * dp).toInt(), 1f
            ).apply { marginEnd = (8 * dp).toInt() }
        }

        btnRerun = Button(this).apply {
            text     = "🔄  RE-RUN"
            textSize = 13f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.parseColor("#FF6B35"))
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(Color.TRANSPARENT)
                setStroke((2 * dp).toInt(), Color.parseColor("#FF6B35"))
                cornerRadius = 4 * dp
            }
            isEnabled = false
            alpha     = 0.4f
            layoutParams = LinearLayout.LayoutParams(0, (52 * dp).toInt(), 0.7f)
        }

        btnRow.addView(btnCamera)
        btnRow.addView(btnPick)
        btnRow.addView(btnRerun)
        root.addView(btnRow, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ))

        setContentView(root)

        // ── Listeners ────────────────────────────────────────────────────
        btnCamera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                launchCamera()
            } else {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    CAMERA_PERM_REQ
                )
            }
        }

        btnPick.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK).apply { type = "image/*" }
            startActivityForResult(intent, PICK_IMAGE_REQ)
        }

        btnRerun.setOnClickListener {
            currentBitmap?.let { runInference(it) }
        }

        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                confThreshold = THRESHOLD_MIN + progress / 100f
                tvThreshold.text = "%.2f".format(confThreshold)
                editThreshold.setText("%.2f".format(confThreshold))
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {
                // Re-run on release if image is loaded
                currentBitmap?.let { runInference(it) }
            }
        })

        btnApply.setOnClickListener {
            val v = editThreshold.text.toString().toFloatOrNull()
            if (v == null || v < THRESHOLD_MIN || v > THRESHOLD_MAX) {
                Toast.makeText(this, "Enter value between 0.01–0.99", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            confThreshold = v
            tvThreshold.text = "%.2f".format(v)
            seekBar.progress = ((v - THRESHOLD_MIN) * 100).toInt()
            val imm = getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
            imm.hideSoftInputFromWindow(editThreshold.windowToken, 0)
            currentBitmap?.let { runInference(it) }
        }
    }

    // ─────────────────────────────────────────
    //  Launch camera with FileProvider URI
    // ─────────────────────────────────────────
    private fun launchCamera() {
        try {
            // externalCacheDir can be null if external storage is unavailable — fall back to internal cache
            val cacheDir = externalCacheDir ?: cacheDir
            val tmpFile = File(cacheDir, "camera_capture_${System.currentTimeMillis()}.jpg")
            val uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                tmpFile
            )
            photoUri = uri
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
                putExtra(MediaStore.EXTRA_OUTPUT, uri)
            }
            startActivityForResult(intent, CAMERA_REQ)
        } catch (e: Exception) {
            Toast.makeText(this, "Cannot open camera: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // ─────────────────────────────────────────
    //  Runtime permission result
    // ─────────────────────────────────────────
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERM_REQ) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCamera()
            } else {
                Toast.makeText(this, "Camera permission is required to use this feature",
                    Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ─────────────────────────────────────────
    //  Image picker / camera result
    // ─────────────────────────────────────────
    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // ── Gallery pick ─────────────────────────────────────────────────
        if (requestCode == PICK_IMAGE_REQ && resultCode == Activity.RESULT_OK) {
            val uri: Uri = data?.data ?: return
            try {
                // Use decodeStream instead of deprecated MediaStore.getBitmap to avoid OOM on large images
                val bmp = contentResolver.openInputStream(uri)?.use {
                    android.graphics.BitmapFactory.decodeStream(it)
                } ?: throw Exception("Could not open image stream")
                loadAndDetect(bmp)
            } catch (e: Exception) {
                Toast.makeText(this, "Could not load image: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

        // ── Camera capture ───────────────────────────────────────────────
        if (requestCode == CAMERA_REQ && resultCode == Activity.RESULT_OK) {
            val uri = photoUri ?: return
            try {
                // Use decodeStream instead of deprecated MediaStore.getBitmap to avoid OOM on large images
                val bmp = contentResolver.openInputStream(uri)?.use {
                    android.graphics.BitmapFactory.decodeStream(it)
                } ?: throw Exception("Could not open camera photo stream")
                loadAndDetect(bmp)
            } catch (e: Exception) {
                Toast.makeText(this, "Could not load captured photo: ${e.message}",
                    Toast.LENGTH_SHORT).show()
            }
        }
    }

    /** Shared helper: display bitmap + kick off inference. */
    private fun loadAndDetect(bmp: android.graphics.Bitmap) {
        currentBitmap = bmp
        imageView.setImageBitmap(bmp)
        imageContainer.findViewWithTag<View>("placeholder")?.visibility = View.GONE
        btnRerun.isEnabled = true
        btnRerun.alpha     = 1f
        overlayView.clearDetections()
        runInference(bmp)
    }

    // ─────────────────────────────────────────
    //  Model / labels
    // ─────────────────────────────────────────
    private fun loadModel() {
        try {
            val fd = assets.openFd(MODEL_FILE)
            val buf: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            interpreter = Interpreter(buf, Interpreter.Options().apply { numThreads = 4 })
            Log.i(TAG, "Model OK")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}")
        }
    }

    private fun loadLabels() {
        try {
            labels = assets.open(LABELS_FILE).bufferedReader()
                .readLines().filter { it.isNotBlank() }
        } catch (_: Exception) {}
    }

    // ─────────────────────────────────────────
    //  Inference on a static bitmap
    // ─────────────────────────────────────────
    private fun runInference(originalBitmap: Bitmap) {
        val interp = interpreter ?: run {
            tvStatus.text = "Model not loaded"
            return
        }

        runOnUiThread {
            progressBar.visibility = View.VISIBLE
            tvStatus.text          = "Running detection…"
            overlayView.clearDetections()
        }

        inferenceHandler.post {
            // Scale to 640×640 for the model
            val scaled = Bitmap.createScaledBitmap(
                originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true
            )

            val buf = ByteBuffer
                .allocateDirect(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 * 4)
                .order(ByteOrder.nativeOrder())
            val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
            scaled.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            for (p in pixels) {
                buf.putFloat(((p shr 16) and 0xFF) / 255f)
                buf.putFloat(((p shr 8)  and 0xFF) / 255f)
                buf.putFloat( (p         and 0xFF) / 255f)
            }
            buf.rewind()

            val out = Array(1) { Array(NUM_CHANNELS) { FloatArray(NUM_PREDS) } }
            try { interp.run(buf, out) }
            catch (e: Exception) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    tvStatus.text = "Inference error: ${e.message}"
                }
                return@post
            }

            val raw  = out[0]
            val dets = mutableListOf<PhotoDetection>()
            val threshold = confThreshold

            for (p in 0 until NUM_PREDS) {
                var bc = 0; var bs = 0f
                for (c in 0 until NUM_CLASSES) {
                    val s = raw[4 + c][p]; if (s > bs) { bs = s; bc = c }
                }
                if (bs < threshold) continue

                val rawCx = raw[0][p]; val rawCy = raw[1][p]
                val rawW  = raw[2][p]; val rawH  = raw[3][p]

                val cx: Float; val cy: Float; val bw: Float; val bh: Float
                if (rawCx > 1f || rawCy > 1f || rawW > 1f || rawH > 1f) {
                    cx = rawCx / MODEL_INPUT_SIZE; cy = rawCy / MODEL_INPUT_SIZE
                    bw = rawW  / MODEL_INPUT_SIZE; bh = rawH  / MODEL_INPUT_SIZE
                } else { cx = rawCx; cy = rawCy; bw = rawW; bh = rawH }

                val left   = (cx - bw / 2f).coerceIn(0f, 1f)
                val top    = (cy - bh / 2f).coerceIn(0f, 1f)
                val right  = (cx + bw / 2f).coerceIn(0f, 1f)
                val bottom = (cy + bh / 2f).coerceIn(0f, 1f)

                val label = if (labels.isNotEmpty() && bc < labels.size) labels[bc] else "cls$bc"
                dets.add(PhotoDetection(RectF(left, top, right, bottom), label, bs))
            }

            val kept = nms(dets)

            runOnUiThread {
                progressBar.visibility = View.GONE
                tvStatus.text = if (kept.isEmpty())
                    "No detections above threshold ${"%.2f".format(threshold)}"
                else
                    "${kept.size} detection${if (kept.size == 1) "" else "s"}  ·  threshold ${"%.2f".format(threshold)}"

                // Pass the original bitmap dimensions so overlay can map correctly
                overlayView.setDetections(kept, originalBitmap, imageView)
            }
        }
    }

    // ─────────────────────────────────────────
    //  NMS
    // ─────────────────────────────────────────
    private fun nms(dets: List<PhotoDetection>): List<PhotoDetection> {
        val sorted = dets.sortedByDescending { it.confidence }
        val sup    = BooleanArray(sorted.size)
        val kept   = mutableListOf<PhotoDetection>()
        for (i in sorted.indices) {
            if (sup[i]) continue
            kept.add(sorted[i])
            for (j in i + 1 until sorted.size)
                if (!sup[j] && iou(sorted[i].rect, sorted[j].rect) > NMS_IOU) sup[j] = true
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iL = maxOf(a.left, b.left);   val iT = maxOf(a.top, b.top)
        val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
        val inter = maxOf(0f, iR - iL) * maxOf(0f, iB - iT)
        if (inter == 0f) return 0f
        return inter / ((a.right-a.left)*(a.bottom-a.top) +
                (b.right-b.left)*(b.bottom-b.top) - inter)
    }
}

// ─────────────────────────────────────────────
//  Data
// ─────────────────────────────────────────────
data class PhotoDetection(
    val rect: RectF,
    val label: String,
    val confidence: Float
)

// ─────────────────────────────────────────────
//  PhotoOverlayView
//
//  Draws boxes over the exact area the ImageView
//  displays the bitmap (respecting FIT_CENTER).
//  Labels are placed far from the boxes (perimeter),
//  connected by colour-matched elbow pointer lines.
// ─────────────────────────────────────────────
class PhotoOverlayView(context: android.content.Context) : android.view.View(context) {

    private var detections: List<PhotoDetection> = emptyList()
    private var imageRect = RectF()   // area the image occupies inside this view

    private val labelColors = mapOf(
        "brace correct"        to 0xFF00FF00.toInt(),
        "brace incorrect"      to 0xFFFF4444.toInt(),
        "central incisor"      to 0xFF00FFCC.toInt(),
        "lateral incisor"      to 0xFFFF33AA.toInt(),
        "ci bracket correct"   to 0xFF44FF88.toInt(),
        "ci bracket incorrect" to 0xFFFF6B35.toInt(),
        "canine"               to 0xFFFFFF00.toInt(),
        "molar"                to 0xFFDD00FF.toInt(),
        "bracket"              to 0xFF00CCFF.toInt()
    )
    private val defaultColor = 0xFFFFFFFF.toInt()

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
        strokeCap = Paint.Cap.ROUND
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 26f          // medium — user preference
        typeface = Typeface.DEFAULT_BOLD
        color    = Color.WHITE
    }

    fun clearDetections() {
        detections = emptyList(); invalidate()
    }

    /** Call after inference with the original bitmap and the ImageView it is displayed in. */
    fun setDetections(dets: List<PhotoDetection>, bitmap: Bitmap, iv: ImageView) {
        detections = dets

        // Compute where FIT_CENTER places the bitmap inside the ImageView
        val vw = iv.width.toFloat();  val vh = iv.height.toFloat()
        val bw = bitmap.width.toFloat(); val bh = bitmap.height.toFloat()
        val scale = minOf(vw / bw, vh / bh)
        val scaledW = bw * scale;  val scaledH = bh * scale
        val offsetX = (vw - scaledW) / 2f
        val offsetY = (vh - scaledH) / 2f

        // imageRect is relative to the ImageView top-left.
        // Since overlayView has the same parent/size as imageView, same offset applies.
        imageRect = RectF(offsetX, offsetY, offsetX + scaledW, offsetY + scaledH)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty() || imageRect.isEmpty) return

        val iw  = imageRect.width()
        val ih  = imageRect.height()
        val ox  = imageRect.left
        val oy  = imageRect.top
        val W   = width.toFloat()
        val H   = height.toFloat()

        // ── Sizing ──────────────────────────────────────────────────────────
        val labelSz = (minOf(iw, ih) * 0.036f).coerceIn(20f, 30f)
        textPaint.textSize = labelSz
        val pad   = 10f
        val pillH = labelSz + pad * 2f

        // ── Build label objects ──────────────────────────────────────────────
        // Local mutable holder
        class Lbl(
            val bL: Float, val bT: Float, val bR: Float, val bB: Float,
            val bcx: Float, val bcy: Float,
            val pw: Float, val txt: String, val col: Int
        ) { var lx = 0f; var ly = 0f }

        val ls = detections.map { det ->
            val c   = labelColors[det.label] ?: defaultColor
            val bL  = ox + det.rect.left   * iw;  val bT = oy + det.rect.top    * ih
            val bR  = ox + det.rect.right  * iw;  val bB = oy + det.rect.bottom * ih
            val bcx = (bL + bR) / 2f;             val bcy = (bT + bB) / 2f

            val isBrace = det.label.contains("brace", ignoreCase = true) ||
                          det.label.contains("bracket", ignoreCase = true)

            val txt = if (isBrace) {
                val suffix = if (det.label.contains("incorrect", ignoreCase = true)) "Incorrect" else "Correct"
                "Braces $suffix  ${"%.0f".format(det.confidence * 100)}%"
            } else {
                ""
            }
            Lbl(bL, bT, bR, bB, bcx, bcy, textPaint.measureText(txt) + pad * 2f, txt, c)
        }

        // ── Sort by box cx → split into TOP group and BOTTOM group ───────────
        // Placing the left-half boxes' labels ABOVE and the right-half BELOW
        // keeps label-x ≈ box-cx in each zone → no line crossings.
        val sorted = ls.sortedBy { it.bcx }
        val nTop   = (sorted.size + 1) / 2
        val topGrp = sorted.take(nTop)
        val botGrp = sorted.drop(nTop)

        // Interleave each group into two rows so wide labels don't pile up
        val topRow1 = topGrp.filterIndexed { i, _ -> i % 2 == 0 }
        val topRow2 = topGrp.filterIndexed { i, _ -> i % 2 == 1 }
        val botRow1 = botGrp.filterIndexed { i, _ -> i % 2 == 0 }
        val botRow2 = botGrp.filterIndexed { i, _ -> i % 2 == 1 }

        // Y positions: place labels in the black bands above/below the image
        val hH     = pillH / 2f
        val topY1  = (oy - hH - 12f).coerceIn(hH + 4f, oy + pillH)
        val topY2  = (topY1 - pillH - 10f).coerceIn(hH + 4f, topY1 - pillH)
        val botY1  = (oy + ih + hH + 12f).coerceIn(oy + ih - pillH, H - hH - 4f)
        val botY2  = (botY1 + pillH + 10f).coerceIn(botY1 + pillH, H - hH - 4f)

        topRow1.forEach { l -> l.lx = l.bcx; l.ly = topY1 }
        topRow2.forEach { l -> l.lx = l.bcx; l.ly = topY2 }
        botRow1.forEach { l -> l.lx = l.bcx; l.ly = botY1 }
        botRow2.forEach { l -> l.lx = l.bcx; l.ly = botY2 }

        // ── Horizontal repulsion within each row ─────────────────────────────
        listOf(topRow1, topRow2, botRow1, botRow2).forEach { row ->
            repeat(20) {
                for (i in row.indices) for (j in i + 1 until row.size) {
                    val a = row[i]; val b = row[j]
                    val dx   = b.lx - a.lx
                    val need = (a.pw + b.pw) / 2f + 8f
                    if (Math.abs(dx) < need) {
                        val push = (need - Math.abs(dx)) / 2f * if (dx >= 0f) 1f else -1f
                        a.lx = (a.lx - push).coerceIn(a.pw / 2f + 2f, W - a.pw / 2f - 2f)
                        b.lx = (b.lx + push).coerceIn(b.pw / 2f + 2f, W - b.pw / 2f - 2f)
                    }
                }
            }
        }

        // ── Pass 1: Bounding boxes — thicker so colour is clearly visible ────
        ls.forEach { l ->
            boxPaint.color = l.col;  boxPaint.strokeWidth = 6f
            canvas.drawRoundRect(l.bL, l.bT, l.bR, l.bB, 10f, 10f, boxPaint)
        }

        // ── Pass 2: Lines box-centre → label-centre ──────────────────────────
        ls.forEach { l ->
            if (l.txt.isNotEmpty()) {
                linePaint.color = l.col
                canvas.drawLine(l.bcx, l.bcy, l.lx, l.ly, linePaint)
                fillPaint.color = l.col
                canvas.drawCircle(l.bcx, l.bcy, 5f, fillPaint)
            }
        }

        // ── Pass 3: Pills — background matches bounding box colour ───────────
        ls.forEach { l ->
            if (l.txt.isNotEmpty()) {
                val pL = l.lx - l.pw / 2f;  val pR = l.lx + l.pw / 2f
                val pT = l.ly - pillH / 2f; val pB = l.ly + pillH / 2f
                // Same colour as the box at ~80 % opacity — clearly identifiable
                fillPaint.color = Color.argb(0xCC, Color.red(l.col), Color.green(l.col), Color.blue(l.col))
                canvas.drawRoundRect(pL, pT, pR, pB, 8f, 8f, fillPaint)
                textPaint.color = Color.WHITE
                canvas.drawText(l.txt, pL + pad, pB - pad, textPaint)
            }
        }
    }
}
