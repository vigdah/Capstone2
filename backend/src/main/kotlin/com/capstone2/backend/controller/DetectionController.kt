package com.capstone2.backend.controller

import com.capstone2.backend.model.DetectionRequest
import com.capstone2.backend.model.DetectionResponse
import com.capstone2.backend.service.DetectionService
import jakarta.validation.Valid
import org.springframework.http.HttpStatus
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.ResponseStatus
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/detections")
class DetectionController(private val detectionService: DetectionService) {

    /**
     * POST /api/detections
     *
     * Android client posts a detection result after each inference cycle.
     * Body: DetectionRequest JSON
     * Response: DetectionResponse with assigned Firestore document ID
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    fun recordDetection(@Valid @RequestBody req: DetectionRequest): DetectionResponse {
        return detectionService.recordDetection(req)
    }

    /**
     * GET /api/detections?userId={userId}
     *
     * Retrieve detection history for a user (latest 100).
     */
    @GetMapping
    fun getDetections(@RequestParam userId: String): List<DetectionResponse> {
        return detectionService.getDetections(userId)
    }
}
