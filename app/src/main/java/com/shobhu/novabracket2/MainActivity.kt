package com.shobhu.novabracket2

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.*
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean

data class Detection(
    val rect: RectF,
    val label: String,
    val confidence: Float
)

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG              = "YOLODetector"
        private const val CAM_PERM_REQ     = 100
        private const val MODEL_FILE       = "best_float16.tflite"
        private const val LABELS_FILE      = "classes.txt"
        private const val MODEL_INPUT_SIZE = 640
        private const val NUM_CHANNELS = 11   // matches your model [1, 11, 8400]
        private const val NUM_PREDS    = 8400
        private const val NUM_CLASSES  = NUM_CHANNELS - 4   // = 7 classes
        private const val NMS_IOU          = 0.45f

        private const val THRESHOLD_MIN    = 0.01f
        private const val THRESHOLD_MAX    = 0.99f
        private const val THRESHOLD_DEF    = 0.25f
    }

    // ── Views ──
    private lateinit var rootLayout:     FrameLayout
    private lateinit var textureView:    TextureView
    private lateinit var overlayView:    OverlayView
    private lateinit var thresholdPanel: LinearLayout
    private lateinit var seekBar:        SeekBar
    private lateinit var editThreshold:  EditText
    private lateinit var btnToggle:      ImageButton

    // ── Camera ──
    private var cameraDevice:   CameraDevice?         = null
    private var captureSession: CameraCaptureSession? = null
    private lateinit var cameraHandler:    Handler
    private lateinit var cameraThread:     HandlerThread
    private lateinit var inferenceHandler: Handler
    private lateinit var inferenceThread:  HandlerThread

    // ── Model ──
    private var interpreter: Interpreter? = null
    private var labels: List<String>      = emptyList()
    private val isInferring               = AtomicBoolean(false)

    @Volatile private var confThreshold   = THRESHOLD_DEF

    private var lastFrameTime = System.currentTimeMillis()
    private var fps           = 0f

    // ─────────────────────────────────────────
    //  Lifecycle
    // ─────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )

        buildUI()

        cameraThread     = HandlerThread("CameraThread").also    { it.start() }
        cameraHandler    = Handler(cameraThread.looper)
        inferenceThread  = HandlerThread("InferenceThread").also { it.start() }
        inferenceHandler = Handler(inferenceThread.looper)

        loadModel()
        loadLabels()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA), CAM_PERM_REQ
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        closeCamera()
        cameraThread.quitSafely()
        inferenceThread.quitSafely()
        interpreter?.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAM_PERM_REQ &&
            grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }

    // ─────────────────────────────────────────
    //  UI
    // ─────────────────────────────────────────
    @SuppressLint("SetTextI18n")
    private fun buildUI() {
        val dp = resources.displayMetrics.density

        rootLayout  = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        textureView = TextureView(this)
        overlayView = OverlayView(this)

        rootLayout.addView(textureView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT, Gravity.CENTER
        ))
        rootLayout.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT, Gravity.CENTER
        ))

        // ── Threshold panel ──────────────────────────────────────────────
        thresholdPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xEE111111.toInt())
            setPadding(
                (16 * dp).toInt(), (12 * dp).toInt(),
                (16 * dp).toInt(), (12 * dp).toInt()
            )
        }

        // Title row
        val titleRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER_VERTICAL
        }

        val titleText = TextView(this).apply {
            text      = "Confidence Threshold"
            textSize  = 13f
            setTextColor(0xFFCCCCCC.toInt())
            typeface  = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        // Current value badge
        val valueBadge = TextView(this).apply {
            text      = "${"%.2f".format(confThreshold)}"
            textSize  = 15f
            setTextColor(0xFF00FFCC.toInt())
            typeface  = Typeface.MONOSPACE
        }

        titleRow.addView(titleText)
        titleRow.addView(valueBadge)

        // SeekBar (0-100 maps to 0.01-0.99)
        seekBar = SeekBar(this).apply {
            max      = 98   // 0-98 → 0.01-0.99
            progress = ((confThreshold - THRESHOLD_MIN) * 100).toInt()
            progressTintList = android.content.res.ColorStateList.valueOf(0xFF00FFCC.toInt())
            thumbTintList    = android.content.res.ColorStateList.valueOf(0xFF00FFCC.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (6 * dp).toInt() }
        }

        // Input row: text field + Apply button
        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (8 * dp).toInt() }
        }

        editThreshold = EditText(this).apply {
            setText("${"%.2f".format(confThreshold)}")
            hint          = "0.01 – 0.99"
            inputType     = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            textSize      = 14f
            setTextColor(Color.WHITE)
            setHintTextColor(0xFF666666.toInt())
            setBackgroundColor(0xFF222222.toInt())
            setPadding(
                (10 * dp).toInt(), (8 * dp).toInt(),
                (10 * dp).toInt(), (8 * dp).toInt()
            )
            imeOptions = EditorInfo.IME_ACTION_DONE
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            ).apply { marginEnd = (8 * dp).toInt() }
        }

        val btnApply = Button(this).apply {
            text      = "Apply"
            textSize  = 13f
            setTextColor(Color.BLACK)
            setBackgroundColor(0xFF00FFCC.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        inputRow.addView(editThreshold)
        inputRow.addView(btnApply)

        // Range labels
        val rangeRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (2 * dp).toInt() }
        }
        val lblLow = TextView(this).apply {
            text = "0.01 (more detections)"
            textSize = 10f
            setTextColor(0xFF888888.toInt())
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val lblHigh = TextView(this).apply {
            text = "0.99 (fewer, confident)"
            textSize = 10f
            setTextColor(0xFF888888.toInt())
            gravity = Gravity.END
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        rangeRow.addView(lblLow)
        rangeRow.addView(lblHigh)

        thresholdPanel.addView(titleRow)
        thresholdPanel.addView(seekBar)
        thresholdPanel.addView(rangeRow)
        thresholdPanel.addView(inputRow)

        // Panel sits at bottom of screen
        val panelLp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        )
        rootLayout.addView(thresholdPanel, panelLp)

        // ── Toggle button (⚙ gear icon) ─────────────────────────────────
        btnToggle = ImageButton(this).apply {
            setImageResource(android.R.drawable.ic_menu_manage)
            setBackgroundColor(0xDD000000.toInt())
            setPadding(
                (12 * dp).toInt(), (12 * dp).toInt(),
                (12 * dp).toInt(), (12 * dp).toInt()
            )
            colorFilter = PorterDuffColorFilter(0xFF00FFCC.toInt(), PorterDuff.Mode.SRC_IN)
        }
        val toggleLp = FrameLayout.LayoutParams(
            (52 * dp).toInt(), (52 * dp).toInt(),
            Gravity.BOTTOM or Gravity.END
        ).apply {
            bottomMargin = (16 * dp).toInt()
            marginEnd    = (16 * dp).toInt()
        }
        rootLayout.addView(btnToggle, toggleLp)

        setContentView(rootLayout)

        // ── Listeners ────────────────────────────────────────────────────

        // Seekbar → update threshold + edit field + badge
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                val v = THRESHOLD_MIN + progress / 100f
                confThreshold = v
                valueBadge.text = "%.2f".format(v)
                // Update edit field without triggering its own listener
                editThreshold.removeTextChangedListener(editWatcher)
                editThreshold.setText("%.2f".format(v))
                editThreshold.addTextChangedListener(editWatcher)
                overlayView.threshold = v
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })

        // EditText → update threshold + seekbar + badge on Apply
        btnApply.setOnClickListener {
            applyEditValue(valueBadge)
            hideKeyboard()
        }
        editThreshold.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                applyEditValue(valueBadge)
                hideKeyboard()
                true
            } else false
        }

        // Toggle panel visibility
        thresholdPanel.visibility = View.GONE  // hidden by default
        btnToggle.setOnClickListener {
            thresholdPanel.visibility =
                if (thresholdPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
    }

    private val editWatcher = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
        override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
        override fun afterTextChanged(s: Editable?) {}
    }

    @SuppressLint("SetTextI18n")
    private fun applyEditValue(badge: TextView) {
        val v = editThreshold.text.toString().toFloatOrNull()
        if (v == null || v < THRESHOLD_MIN || v > THRESHOLD_MAX) {
            Toast.makeText(this,
                "Enter a value between $THRESHOLD_MIN and $THRESHOLD_MAX",
                Toast.LENGTH_SHORT).show()
            return
        }
        confThreshold = v
        overlayView.threshold = v
        badge.text = "%.2f".format(v)
        seekBar.progress = ((v - THRESHOLD_MIN) * 100).toInt()
    }

    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(editThreshold.windowToken, 0)
    }

    // ─────────────────────────────────────────
    //  Model / labels
    // ─────────────────────────────────────────
    private fun loadModel() {
        try { Log.i(TAG, "Assets: ${assets.list("")?.joinToString()}") }
        catch (e: Exception) { Log.e(TAG, "list assets: ${e.message}") }
        try {
            val fd = assets.openFd(MODEL_FILE)
            val buf: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            interpreter = Interpreter(buf, Interpreter.Options().apply { numThreads = 4 })
            Log.i(TAG, "Model OK  in=${interpreter!!.getInputTensor(0).shape().toList()}" +
                    "  out=${interpreter!!.getOutputTensor(0).shape().toList()}")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}")
            overlayView.modelStatus = "Model FAILED"
        }
    }

    private fun loadLabels() {
        try {
            labels = assets.open(LABELS_FILE).bufferedReader()
                .readLines().filter { it.isNotBlank() }
            Log.i(TAG, "Labels: ${labels.size}")
        } catch (_: Exception) { Log.w(TAG, "No labels.txt") }
    }

    // ─────────────────────────────────────────
    //  Camera
    // ─────────────────────────────────────────
    private fun startCamera() {
        if (textureView.isAvailable) openCamera()
        else textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) = openCamera()
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) = grabAndInfer()
        }
    }

    private fun grabAndInfer() {
        if (!isInferring.compareAndSet(false, true)) return
        val bmp = textureView.getBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE) ?: run {
            isInferring.set(false); return
        }
        inferenceHandler.post {
            try { runInference(bmp) } finally { isInferring.set(false) }
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) = grabAndInfer()
        }

        val manager  = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = manager.cameraIdList.firstOrNull { id ->
            manager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        } ?: manager.cameraIdList[0]

        val map = manager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

        val screenW = resources.displayMetrics.widthPixels
        val screenH = resources.displayMetrics.heightPixels

        val previewSize: Size = map.getOutputSizes(SurfaceTexture::class.java)
            .filter { it.height <= screenW && it.width <= screenH }
            .maxByOrNull { it.width.toLong() * it.height }
            ?: map.getOutputSizes(SurfaceTexture::class.java)[0]

        val dispW  = previewSize.height.toFloat()
        val dispH  = previewSize.width.toFloat()
        val scale  = minOf(screenW / dispW, screenH / dispH)
        val finalW = (dispW * scale).toInt()
        val finalH = (dispH * scale).toInt()

        runOnUiThread {
            textureView.layoutParams = FrameLayout.LayoutParams(finalW, finalH, Gravity.CENTER)
            overlayView.layoutParams = FrameLayout.LayoutParams(finalW, finalH, Gravity.CENTER)
        }

        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(cam: CameraDevice) {
                cameraDevice = cam; createCaptureSession(previewSize)
            }
            override fun onDisconnected(cam: CameraDevice) { cam.close() }
            override fun onError(cam: CameraDevice, e: Int) { cam.close() }
        }, cameraHandler)
    }

    private fun createCaptureSession(previewSize: Size) {
        val st = textureView.surfaceTexture ?: return
        st.setDefaultBufferSize(previewSize.width, previewSize.height)
        val previewSurface = Surface(st)
        cameraDevice?.createCaptureSession(listOf(previewSurface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    val req = cameraDevice!!
                        .createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(previewSurface)
                            set(CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                        }.build()
                    session.setRepeatingRequest(req, null, cameraHandler)
                }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    Log.e(TAG, "Session config failed")
                }
            }, cameraHandler)
    }

    private fun closeCamera() {
        captureSession?.close(); captureSession = null
        cameraDevice?.close();   cameraDevice   = null
    }

    // ─────────────────────────────────────────
    //  Inference
    // ─────────────────────────────────────────
    private fun runInference(bitmap: Bitmap) {
        val interp    = interpreter ?: return
        val threshold = confThreshold   // snapshot volatile

        val now = System.currentTimeMillis()
        fps = 1000f / (now - lastFrameTime).coerceAtLeast(1)
        lastFrameTime = now

        val buf = ByteBuffer
            .allocateDirect(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        bitmap.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        for (p in pixels) {
            buf.putFloat(((p shr 16) and 0xFF) / 255f)
            buf.putFloat(((p shr 8)  and 0xFF) / 255f)
            buf.putFloat( (p         and 0xFF) / 255f)
        }
        buf.rewind()

        val out = Array(1) { Array(NUM_CHANNELS) { FloatArray(NUM_PREDS) } }
        try { interp.run(buf, out) }
        catch (e: Exception) {
            Log.e(TAG, "run() failed: ${e.message}")
            overlayView.post { overlayView.modelStatus = "Inference error" }
            return
        }

        val raw  = out[0]
        val dets = mutableListOf<Detection>()

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
            dets.add(Detection(RectF(left, top, right, bottom), label, bs))
        }

        val kept = nms(dets)
        overlayView.post {
            overlayView.detections  = kept
            overlayView.fps         = fps
            overlayView.modelStatus = if (kept.isEmpty()) "Running — no det" else "Running ✓"
            overlayView.invalidate()
        }
    }

    // ─────────────────────────────────────────
    //  NMS
    // ─────────────────────────────────────────
    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.confidence }
        val sup    = BooleanArray(sorted.size)
        val kept   = mutableListOf<Detection>()
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

    // ═════════════════════════════════════════
    //  OverlayView  —  callout / leader-line style
    //  Labels orbit around the view perimeter and
    //  connect to boxes via elbow pointer lines.
    //  Box, line, and label are all the same colour.
    // ═════════════════════════════════════════
    inner class OverlayView(context: Context) : View(context) {

        var detections:  List<Detection> = emptyList()
        var fps:         Float           = 0f
        var modelStatus: String          = "Initializing…"
        var threshold:   Float           = THRESHOLD_DEF

        private val labelColors = mapOf(
            "brace correct"        to 0xFF00FF00.toInt(),
            "brace incorrect"      to 0xFFFF4444.toInt(),
            "central incisor"      to 0xFF00FFCC.toInt(),
            "lateral incisor"      to 0xFFFF33AA.toInt(),
            "ci bracket correct"   to 0xFF44FF88.toInt(),
            "ci bracket incorrect" to 0xFFFF6B35.toInt(),
            "canine"               to 0xFFFFEE44.toInt(),
            "molar"                to 0xFFDD44FF.toInt(),
            "bracket"              to 0xFF44CCFF.toInt()
        )
        private val defaultColor = 0xFFFFFFFF.toInt()

        private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f
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
            textSize = 26f          // medium size
            typeface = Typeface.DEFAULT_BOLD
            color    = Color.WHITE
        }
        private val hudBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xDD111111.toInt(); style = Paint.Style.FILL
        }
        private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = 34f; typeface = Typeface.MONOSPACE
            setShadowLayer(4f, 1f, 1f, Color.BLACK)
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val W = width.toFloat()
            val H = height.toFloat()

            if (detections.isNotEmpty()) {

                // ── Sizing ─────────────────────────────────────────────────────
                val labelSz = (minOf(W, H) * 0.036f).coerceIn(20f, 30f)
                textPaint.textSize = labelSz
                val pad   = 10f
                val pillH = labelSz + pad * 2f

                // ── Build label objects ─────────────────────────────────────────
                class Lbl(
                    val bL: Float, val bT: Float, val bR: Float, val bB: Float,
                    val bcx: Float, val bcy: Float,
                    val pw: Float, val txt: String, val col: Int
                ) { var lx = 0f; var ly = 0f }

                val ls = detections.map { det ->
                    val c   = labelColors[det.label] ?: defaultColor
                    val bL  = det.rect.left   * W;  val bT = det.rect.top    * H
                    val bR  = det.rect.right  * W;  val bB = det.rect.bottom * H
                    val bcx = (bL + bR) / 2f;       val bcy = (bT + bB) / 2f

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

                // ── Sort by box cx → top (left half) and bottom (right half) ──
                val sorted = ls.sortedBy { it.bcx }
                val nTop   = (sorted.size + 1) / 2
                val topGrp = sorted.take(nTop)
                val botGrp = sorted.drop(nTop)

                // Two interleaved rows per zone so wide labels don't crush together
                val topRow1 = topGrp.filterIndexed { i, _ -> i % 2 == 0 }
                val topRow2 = topGrp.filterIndexed { i, _ -> i % 2 == 1 }
                val botRow1 = botGrp.filterIndexed { i, _ -> i % 2 == 0 }
                val botRow2 = botGrp.filterIndexed { i, _ -> i % 2 == 1 }

                val hH    = pillH / 2f
                val topY1 = hH + 6f
                val topY2 = topY1 + pillH + 8f
                val botY1 = H - hH - 6f
                val botY2 = botY1 - pillH - 8f

                topRow1.forEach { l -> l.lx = l.bcx; l.ly = topY1 }
                topRow2.forEach { l -> l.lx = l.bcx; l.ly = topY2 }
                botRow1.forEach { l -> l.lx = l.bcx; l.ly = botY1 }
                botRow2.forEach { l -> l.lx = l.bcx; l.ly = botY2 }

                // ── Horizontal repulsion within each row ──────────────────────
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

                // ── Pass 1: Boxes — thicker stroke ─────────────────────────────
                ls.forEach { l ->
                    boxPaint.color = l.col;  boxPaint.strokeWidth = 7f
                    canvas.drawRoundRect(l.bL, l.bT, l.bR, l.bB, 10f, 10f, boxPaint)
                }

                // ── Pass 2: Lines ───────────────────────────────────────────────
                ls.forEach { l ->
                    if (l.txt.isNotEmpty()) {
                        linePaint.color = l.col
                        canvas.drawLine(l.bcx, l.bcy, l.lx, l.ly, linePaint)
                        fillPaint.color = l.col
                        canvas.drawCircle(l.bcx, l.bcy, 6f, fillPaint)
                    }
                }

                // ── Pass 3: Pills — background = box colour at 80 % opacity ────
                ls.forEach { l ->
                    if (l.txt.isNotEmpty()) {
                        val pL = l.lx - l.pw / 2f;  val pR = l.lx + l.pw / 2f
                        val pT = l.ly - pillH / 2f; val pB = l.ly + pillH / 2f
                        fillPaint.color = Color.argb(0xCC, Color.red(l.col), Color.green(l.col), Color.blue(l.col))
                        canvas.drawRoundRect(pL, pT, pR, pB, 8f, 8f, fillPaint)
                        textPaint.color = Color.WHITE
                        canvas.drawText(l.txt, pL + pad, pB - pad, textPaint)
                    }
                }
            }

            // ── HUD ────────────────────────────────────────────────────────────
            val hudLines = listOf(
                Pair("FPS   : ${"%.1f".format(fps)}",        0xFF00FFFF.toInt()),
                Pair("DETS  : ${detections.size}",            0xFF88FF88.toInt()),
                Pair("THRESH: ${"%.2f".format(threshold)}",  0xFFFFCC44.toInt()),
                Pair("MODEL : $modelStatus",
                    if (modelStatus.contains("✓")) 0xFF88FF88.toInt() else 0xFFFFAA33.toInt())
            )
            val lh   = hudPaint.textSize + 14f
            val hudW = hudLines.maxOf { hudPaint.measureText(it.first) } + 32f
            val hudH = lh * hudLines.size + 16f
            canvas.drawRoundRect(16f, 16f, 16f + hudW, 16f + hudH, 12f, 12f, hudBgPaint)
            hudLines.forEachIndexed { i, (text, col) ->
                hudPaint.color = col
                canvas.drawText(text, 28f, 16f + 18f + (i + 1) * lh - 6f, hudPaint)
            }
        }
    }
}