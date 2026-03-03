package com.capstone2.backend.model

import jakarta.validation.constraints.DecimalMax
import jakarta.validation.constraints.DecimalMin
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull

data class DetectionRequest(
    @field:NotBlank val userId: String,
    @field:NotBlank val packageName: String,
    @field:NotBlank val appName: String,
    @field:NotNull
    @field:DecimalMin("0.0")
    @field:DecimalMax("1.0")
    val score: Float,
    @field:NotBlank val label: String,
    val confidence: Float,
    val detectedAtMs: Long = System.currentTimeMillis()
)

data class DetectionResponse(
    val id: String,
    val packageName: String,
    val appName: String,
    val score: Float,
    val label: String,
    val confidence: Float,
    val detectedAtMs: Long
)

data class Alert(
    val id: String = "",
    val userId: String = "",
    val packageName: String = "",
    val appName: String = "",
    val score: Float = 0f,
    val message: String = "",
    val isDismissed: Boolean = false,
    val triggeredAtMs: Long = 0L
)

data class AlertRequest(
    @field:NotBlank val userId: String,
    val alertId: String
)

data class ModelInfo(
    val version: String,
    val downloadUrl: String,
    val sizeBytes: Long,
    val publishedAtMs: Long,
    val minAppVersion: Int = 1
)
