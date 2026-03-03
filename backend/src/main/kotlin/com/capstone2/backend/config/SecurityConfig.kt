package com.capstone2.backend.config

import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.config.http.SessionCreationPolicy
import org.springframework.security.web.SecurityFilterChain
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter
import org.springframework.web.filter.OncePerRequestFilter

/**
 * Simple API key authentication for MVP.
 * In production: replace with Firebase Auth JWT verification.
 */
@Configuration
@EnableWebSecurity
class SecurityConfig(@Value("\${app.api-key}") private val apiKey: String) {

    @Bean
    fun securityFilterChain(http: HttpSecurity): SecurityFilterChain {
        http
            .csrf { it.disable() }
            .sessionManagement { it.sessionCreationPolicy(SessionCreationPolicy.STATELESS) }
            .authorizeHttpRequests { auth ->
                auth
                    .requestMatchers("/actuator/health").permitAll()
                    .anyRequest().authenticated()
            }
            .addFilterBefore(ApiKeyFilter(apiKey), UsernamePasswordAuthenticationFilter::class.java)

        return http.build()
    }
}

class ApiKeyFilter(private val validApiKey: String) : OncePerRequestFilter() {
    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        filterChain: FilterChain
    ) {
        val providedKey = request.getHeader("X-Api-Key")
        if (providedKey == validApiKey) {
            // Mark request as authenticated via security context
            val auth = org.springframework.security.authentication.UsernamePasswordAuthenticationToken(
                "android-client", null, emptyList()
            )
            org.springframework.security.core.context.SecurityContextHolder.getContext().authentication = auth
            filterChain.doFilter(request, response)
        } else {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Invalid or missing API key")
        }
    }
}
