package com.capstone2.backend.service

import com.capstone2.backend.model.DetectionRequest
import com.capstone2.backend.model.DetectionResponse
import com.capstone2.backend.repository.FirestoreRepository
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service

@Service
class DetectionService(private val repo: FirestoreRepository) {

    private val log = LoggerFactory.getLogger(DetectionService::class.java)

    fun recordDetection(req: DetectionRequest): DetectionResponse {
        log.debug("Recording detection: pkg=${req.packageName}, score=${req.score}, label=${req.label}")
        val response = repo.saveDetection(req)

        if (req.score >= HIGH_RISK_THRESHOLD) {
            log.warn("HIGH RISK detection: user=${req.userId}, pkg=${req.packageName}, score=${req.score}")
        }

        return response
    }

    fun getDetections(userId: String): List<DetectionResponse> {
        return repo.getDetections(userId)
    }

    companion object {
        private const val HIGH_RISK_THRESHOLD = 0.7f
    }
}
