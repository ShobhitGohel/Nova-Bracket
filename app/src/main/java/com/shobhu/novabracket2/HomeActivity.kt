package com.shobhu.novabracket2

import android.content.Intent
import android.graphics.*
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.animation.AlphaAnimation
import android.view.animation.Animation
import android.view.animation.DecelerateInterpolator
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )

        val dp = resources.displayMetrics.density

        // ── Root ──────────────────────────────────────────────────────────
        val root = FrameLayout(this).apply { setBackgroundColor(Color.parseColor("#050A0F")) }

        // Animated scan-line canvas background
        val bgCanvas = ScanLineView(this)
        root.addView(bgCanvas, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        ))

        // ── Centre column ─────────────────────────────────────────────────
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity     = Gravity.CENTER_HORIZONTAL
            setPadding((32 * dp).toInt(), 0, (32 * dp).toInt(), 0)
        }
        root.addView(col, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT,
            Gravity.CENTER
        ))

        // ── Logo / icon ───────────────────────────────────────────────────
        val logoView = LogoView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                (120 * dp).toInt(), (120 * dp).toInt()
            ).apply { bottomMargin = (28 * dp).toInt() }
        }
        col.addView(logoView)

        // ── App title ─────────────────────────────────────────────────────
        val title = TextView(this).apply {
            text      = "NOVA BRACKET"
            textSize  = 30f
            typeface  = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.WHITE)
            gravity   = Gravity.CENTER
            letterSpacing = 0.15f
            layoutParams  = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (6 * dp).toInt() }
        }
        col.addView(title)

        val subtitle = TextView(this).apply {
            text     = "AI  ·  OBJECT  DETECTION"
            textSize = 11f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
            setTextColor(Color.parseColor("#00FFCC"))
            gravity  = Gravity.CENTER
            letterSpacing = 0.3f
            layoutParams  = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (56 * dp).toInt() }
        }
        col.addView(subtitle)

        // ── Divider ───────────────────────────────────────────────────────
        val divider = View(this).apply {
            setBackgroundColor(Color.parseColor("#1A2A2A"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, (1 * dp).toInt()
            ).apply { bottomMargin = (48 * dp).toInt() }
        }
        col.addView(divider)

        // ── Live Camera button ────────────────────────────────────────────
        val btnCamera = makeOptionButton(
            dp       = dp,
            icon     = "📷",
            title    = "LIVE CAMERA",
            subtitle = "Real-time detection",
            accent   = "#00FFCC"
        )
        col.addView(btnCamera)

        val spacer = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                1, (20 * dp).toInt()
            )
        }
        col.addView(spacer)

        // ── Upload Photo button ───────────────────────────────────────────
        val btnPhoto = makeOptionButton(
            dp       = dp,
            icon     = "🖼",
            title    = "UPLOAD PHOTO",
            subtitle = "Detect from gallery",
            accent   = "#FF6B35"
        )
        col.addView(btnPhoto)

        // ── Version tag ───────────────────────────────────────────────────
        val version = TextView(this).apply {
            text     = "v1.0  ·  YOLOv8"
            textSize = 10f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
            setTextColor(Color.parseColor("#2A4A4A"))
            gravity  = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (48 * dp).toInt() }
        }
        col.addView(version)

        setContentView(root)
        // statusBarColor must be set AFTER setContentView on Samsung/Android 12+ with Material3 theme
        window.statusBarColor = Color.BLACK

        // ── Fade-in entrance ──────────────────────────────────────────────
        listOf(logoView, title, subtitle, btnCamera, btnPhoto).forEachIndexed { i, v ->
            v.alpha = 0f
            v.animate()
                .alpha(1f)
                .setStartDelay((120L * i))
                .setDuration(500)
                .setInterpolator(DecelerateInterpolator())
                .start()
        }

        // ── Click handlers ────────────────────────────────────────────────
        btnCamera.setOnClickListener {
            it.animate().scaleX(0.95f).scaleY(0.95f).setDuration(80).withEndAction {
                it.animate().scaleX(1f).scaleY(1f).setDuration(80).withEndAction {
                    startActivity(Intent(this, MainActivity::class.java))
                }.start()
            }.start()
        }

        btnPhoto.setOnClickListener {
            it.animate().scaleX(0.95f).scaleY(0.95f).setDuration(80).withEndAction {
                it.animate().scaleX(1f).scaleY(1f).setDuration(80).withEndAction {
                    startActivity(Intent(this, PhotoActivity::class.java))
                }.start()
            }.start()
        }
    }

    // ── Helper: build a styled option card ───────────────────────────────
    private fun makeOptionButton(
        dp: Float, icon: String, title: String, subtitle: String, accent: String
    ): LinearLayout {
        val accentColor = Color.parseColor(accent)

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity     = Gravity.CENTER_VERTICAL
            setBackgroundColor(Color.parseColor("#0D1A1A"))
            setPadding(
                (20 * dp).toInt(), (20 * dp).toInt(),
                (20 * dp).toInt(), (20 * dp).toInt()
            )
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                (80 * dp).toInt()
            )
            // Border via background drawable
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(Color.parseColor("#0D1A1A"))
                setStroke((1.5f * dp).toInt(), accentColor)
                cornerRadius = 12 * dp
            }
        }

        // Icon
        val iconView = TextView(this).apply {
            text     = icon
            textSize = 28f
            gravity  = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                (52 * dp).toInt(), (52 * dp).toInt()
            ).apply { marginEnd = (16 * dp).toInt() }
        }

        // Text column
        val textCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val titleView = TextView(this).apply {
            text     = title
            textSize = 16f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            setTextColor(Color.WHITE)
            letterSpacing = 0.1f
        }
        val subView = TextView(this).apply {
            text     = subtitle
            textSize = 11f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
            setTextColor(Color.parseColor("#607070"))
        }
        textCol.addView(titleView)
        textCol.addView(subView)

        // Arrow
        val arrow = TextView(this).apply {
            text     = "›"
            textSize = 28f
            setTextColor(accentColor)
            gravity  = Gravity.CENTER
        }

        card.addView(iconView)
        card.addView(textCol)
        card.addView(arrow)
        return card
    }

    // ── Animated scan-line background ─────────────────────────────────────
    inner class ScanLineView(context: android.content.Context) : View(context) {
        private val paint = Paint().apply {
            color = Color.parseColor("#0A1F1F")
            strokeWidth = 1f
        }
        private var offset = 0f
        private val handler = android.os.Handler(android.os.Looper.getMainLooper())
        private val ticker = object : Runnable {
            override fun run() { offset = (offset + 0.5f) % 40f; invalidate(); handler.postDelayed(this, 33) }
        }
        override fun onAttachedToWindow() { super.onAttachedToWindow(); handler.post(ticker) }
        override fun onDetachedFromWindow() { super.onDetachedFromWindow(); handler.removeCallbacks(ticker) }
        override fun onDraw(canvas: Canvas) {
            var y = -offset
            while (y < height) { canvas.drawLine(0f, y, width.toFloat(), y, paint); y += 40f }
        }
    }

    // ── Pulsing hexagon logo ───────────────────────────────────────────────
    inner class LogoView(context: android.content.Context) : View(context) {
        private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE; strokeWidth = 3f
            color = Color.parseColor("#00FFCC")
        }
        private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#051515")
        }
        private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color    = Color.parseColor("#00FFCC")
            textSize = 36f
            typeface = Typeface.DEFAULT_BOLD
            textAlign = Paint.Align.CENTER
        }
        private var pulse = 0f
        private val handler = android.os.Handler(android.os.Looper.getMainLooper())
        private val ticker = object : Runnable {
            override fun run() {
                pulse = (pulse + 0.05f) % (2 * Math.PI.toFloat())
                invalidate(); handler.postDelayed(this, 33)
            }
        }
        override fun onAttachedToWindow() { super.onAttachedToWindow(); handler.post(ticker) }
        override fun onDetachedFromWindow() { super.onDetachedFromWindow(); handler.removeCallbacks(ticker) }
        override fun onDraw(canvas: Canvas) {
            val cx = width / 2f; val cy = height / 2f
            val r  = minOf(cx, cy) - 6f
            val pulseR = r + 8f * kotlin.math.sin(pulse)

            // Outer pulsing ring
            ringPaint.alpha = (180 + 75 * kotlin.math.sin(pulse)).toInt()
            canvas.drawCircle(cx, cy, pulseR, ringPaint)

            // Inner circle fill
            canvas.drawCircle(cx, cy, r - 6f, fillPaint)

            // Inner ring
            ringPaint.alpha = 255
            canvas.drawCircle(cx, cy, r - 6f, ringPaint)

            // AI text
            canvas.drawText("AI", cx, cy + textPaint.textSize / 3f, textPaint)
        }
    }
}