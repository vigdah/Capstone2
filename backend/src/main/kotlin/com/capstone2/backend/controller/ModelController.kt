package com.capstone2.backend.controller

import com.capstone2.backend.model.ModelInfo
import com.capstone2.backend.repository.FirestoreRepository
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/model")
class ModelController(private val repo: FirestoreRepository) {

    /**
     * GET /api/model/latest
     *
     * Returns the latest model version info.
     * Android client checks this to determine if a model update is available.
     * If a newer version exists, app downloads and replaces model.onnx in assets.
     */
    @GetMapping("/latest")
    fun getLatestModel(): ResponseEntity<ModelInfo> {
        val info = repo.getLatestModelInfo()
            ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok(info)
    }
}
