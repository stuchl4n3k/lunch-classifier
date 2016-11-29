package net.stuchl4n3k.lunchtime;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * @author petr.stuchlik
 */
@SpringBootApplication
@Slf4j
public class LunchtimeWebApp {

    public static void main(String[] args) throws Exception {
        SpringApplication.run(LunchtimeWebApp.class, args);
    }
}
