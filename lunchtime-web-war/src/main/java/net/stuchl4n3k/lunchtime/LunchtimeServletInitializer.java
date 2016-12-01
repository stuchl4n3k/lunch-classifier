package net.stuchl4n3k.lunchtime;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.web.support.SpringBootServletInitializer;
import org.springframework.web.WebApplicationInitializer;

/**
 * @author petr.stuchlik
 */
@SpringBootApplication
public class LunchtimeServletInitializer extends SpringBootServletInitializer implements WebApplicationInitializer {

    @Override
    protected SpringApplicationBuilder configure(SpringApplicationBuilder application) {
        return application.sources(LunchtimeServletInitializer.class);
    }

    public static void main(String[] args) throws Exception {
        SpringApplication.run(LunchtimeServletInitializer.class, args);
    }

}
